/* PhotoPlay AR —— 主线程：采集 + 渲染
 *
 * 架构：ORB 识别（合并索引单次匹配）与 LK 光流追踪在 cv-worker.js 中执行，
 * 主线程只负责摄像头采集与 WebGL 透视绘制。
 * 渲染直接使用 worker 上报的单应性，不做任何滤波/插值/延迟补偿，
 * 保证视频"粘贴"在照片上的跟随手感。
 * 视频真实画面帧呈现前不绘制（framePresented 门控），避免加载期黑块。
 * 后端只负责静态下发：/api/features(二进制特征库) /videos/{id}
 */
'use strict';

/* ---------------- 常量 ---------------- */
const WORK_W = 640;          // 帧采集统一工作宽度
const MAX_WARM_VIDEOS = 3;   // 视频预热池上限（超出淘汰最久未命中的）
const FULL_PRELOAD_MAX = 4;  // 库内视频数不超过此值时直接全量预热

/* ---------------- DOM ---------------- */
const cameraEl = document.getElementById('camera');
const glCanvas = document.getElementById('gl');
const arVideo = document.getElementById('ar-video');
const soundBtn = document.getElementById('sound-btn');
const hintEl = document.getElementById('scan-hint');
const statusDot = document.getElementById('status-dot');

/* ---------------- 状态 ---------------- */
let worker = null;
let workerReady = false;
let inFlight = false;        // 上一帧仍在 worker 处理中（流控，避免帧堆积）
let library = [];            // {id,title,videoUrl}（仅元数据；特征在 worker 内）
let state = 'boot';          // boot | detect | track
let targetH = null;          // worker 上报的最新单应性（渲染直接使用，无滤波）
let framePresented = false;  // 视频已呈现真实画面帧（此前不绘制，避免黑帧/加载态黑块）

// 帧采集画布
const workCanvas = document.createElement('canvas');
const workCtx = workCanvas.getContext('2d', { willReadFrequently: true });

/* ---------------- 3x3 矩阵（行优先） ---------------- */
function mul3(A, B) {
  const C = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return C;
}
function mat3ToGL(M) { // 行优先 -> GL 列优先
  return new Float32Array([M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]]);
}

/* ---------------- 摄像头 ---------------- */
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    cameraEl.srcObject = stream;
  } catch (e) {
    document.getElementById('fallback-tip').style.display = 'flex';
  }
}

/* ---------------- 帧采集（RGBA，Transferable 零拷贝传给 worker） ---------------- */
function grabRGBA() {
  const vw = cameraEl.videoWidth, vh = cameraEl.videoHeight;
  if (!vw || !vh) return null;
  const scale = Math.min(1, WORK_W / vw);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  workCanvas.width = w; workCanvas.height = h;
  workCtx.drawImage(cameraEl, 0, 0, w, h);
  const imgData = workCtx.getImageData(0, 0, w, h);
  return { buf: imgData.data.buffer, w, h };
}

/* ---------------- 特征库加载（二进制，切片后零拷贝移交 worker） ---------------- */
async function loadLibrary() {
  const buf = await (await fetch('/api/features')).arrayBuffer();
  const dv = new DataView(buf);
  const jlen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jlen)));
  const base = 4 + jlen;
  const items = [];
  const transfers = [];
  for (const it of meta) {
    try {
      const ptsBuf = buf.slice(base + it.pts_off, base + it.pts_off + it.n * 2 * 4);
      const desBuf = buf.slice(base + it.des_off, base + it.des_off + it.n * 32);
      library.push({ id: it.photo_id, title: it.title, videoUrl: it.video_url });
      items.push({ n: it.n, ptsBuf, desBuf });
      transfers.push(ptsBuf, desBuf);
    } catch (e) {
      console.error('特征加载失败', it.photo_id, e);
    }
  }
  worker.postMessage({ type: 'library', items }, transfers);
}

/* ---------------- 视频预热（候选触发 + 小库全量） ---------------- */
// worker 上报疑似候选时后台缓冲其视频；服务端已下发 Cache-Control，
// 预热缓冲可被 arVideo 直接复用，命中即播。预热池有上限，避免抢占带宽。
const warmPool = new Map();  // photo_id -> {el, t}

function warmVideo(item) {
  if (warmPool.has(item.id)) return;
  if (warmPool.size >= MAX_WARM_VIDEOS) {
    // 淘汰最早入池的预热元素，释放其连接与缓冲
    let oldestId = null, oldestT = Infinity;
    for (const [id, w] of warmPool) {
      if (w.t < oldestT) { oldestT = w.t; oldestId = id; }
    }
    const old = warmPool.get(oldestId);
    old.el.removeAttribute('src');
    old.el.load();
    warmPool.delete(oldestId);
  }
  const v = document.createElement('video');
  v.preload = 'auto';
  v.muted = true;
  v.playsInline = true;
  v.src = item.videoUrl;
  v.load();
  warmPool.set(item.id, { el: v, t: performance.now() });
}

// 库很小（视频数 <= FULL_PRELOAD_MAX）时全量预热，代价可忽略、收益最大
function maybePreloadAll() {
  if (library.length <= FULL_PRELOAD_MAX) {
    for (const item of library) warmVideo(item);
  }
}

/* ---------------- WebGL 渲染 ---------------- */
let gl = null, glProg = null, glTex = null, uM = null, uAlpha = null;

function initGL() {
  gl = glCanvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
  if (!gl) { hintEl.textContent = '当前浏览器不支持 WebGL'; return false; }
  const vs = `
    attribute vec2 a_pos;
    uniform mat3 u_M;
    varying vec2 v_uv;
    void main() {
      vec3 p = u_M * vec3(a_pos, 1.0);
      v_uv = a_pos;
      gl_Position = vec4(p.xy, 0.0, p.z);
    }`;
  const fs = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_tex;
    uniform float u_alpha;
    void main() {
      vec4 c = texture2D(u_tex, v_uv);
      gl_FragColor = vec4(c.rgb, c.a * u_alpha);
    }`;
  function shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  }
  glProg = gl.createProgram();
  gl.attachShader(glProg, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(glProg, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(glProg);
  gl.useProgram(glProg);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(glProg, 'a_pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  uM = gl.getUniformLocation(glProg, 'u_M');
  uAlpha = gl.getUniformLocation(glProg, 'u_alpha');
  glTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, glTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // 不启用 UNPACK_FLIP_Y：clip 空间 Y 翻转与纹理 v 翻转相互抵消，
  // 双翻转下纹理按原始方向采样，内容与屏幕坐标一致
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  return true;
}

function resizeGL() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = glCanvas.getBoundingClientRect();
  glCanvas.width = Math.round(rect.width * dpr);
  glCanvas.height = Math.round(rect.height * dpr);
}

function clearGL() {
  if (!gl) return;
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

function drawOverlay(H) {
  if (!gl) return;
  if (!framePresented || arVideo.readyState < 2) return; // 视频真实画面呈现前不绘制（无黑块）
  const vw = cameraEl.videoWidth, vh = cameraEl.videoHeight;
  if (!vw || !vh) return;
  const rect = cameraEl.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const dispScale = Math.max(rect.width / vw, rect.height / vh);
  const offX = rect.left + (rect.width - vw * dispScale) / 2;
  const offY = rect.top + (rect.height - vh * dispScale) / 2;
  const workScale = Math.min(1, WORK_W / vw);
  const k = dispScale / workScale * dpr; // 工作帧像素 -> 画布像素
  // 画布原点与 camera 元素左上对齐（canvas 覆盖整个 stage，camera 也在 stage 内同位）
  const canvasRect = glCanvas.getBoundingClientRect();
  const S = [k, 0, (offX - canvasRect.left) * dpr,
             0, k, (offY - canvasRect.top) * dpr,
             0, 0, 1];
  const W = glCanvas.width, Hh = glCanvas.height;
  const C = [2 / W, 0, -1, 0, -2 / Hh, 1, 0, 0, 1]; // CSS->clip（Y 翻转）
  const M = mul3(C, mul3(S, H));

  gl.viewport(0, 0, W, Hh);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindTexture(gl.TEXTURE_2D, glTex);
  try {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, arVideo);
  } catch (e) { return; }
  gl.uniformMatrix3fv(uM, false, mat3ToGL(M));
  gl.uniform1f(uAlpha, 1.0);
  gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
}

/* ---------------- 渲染循环（直接绘制最新单应性，无滤波，粘贴式跟随） ---------------- */
function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (state !== 'track' || !targetH) return;
  drawOverlay(targetH);
}

/* ---------------- 帧泵（rVFC 对齐帧节奏，setTimeout 兜底） ---------------- */
const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
function schedulePump() {
  if (hasRVFC && cameraEl.srcObject) {
    cameraEl.requestVideoFrameCallback(() => pump());
  } else {
    setTimeout(pump, 33);
  }
}
function pump() {
  if (workerReady && !inFlight &&
      (state === 'detect' || state === 'track') &&
      cameraEl.readyState >= 2) {
    const f = grabRGBA();
    if (f) {
      inFlight = true;
      worker.postMessage({ type: 'frame', buf: f.buf, w: f.w, h: f.h }, [f.buf]);
    }
  }
  schedulePump();
}

/* ---------------- 命中 / 丢失 / 播完 ---------------- */
function onLocked(item, H) {
  state = 'track';
  targetH = H;

  if (arVideo.dataset.photoId !== item.id) {
    arVideo.dataset.photoId = item.id;
    arVideo.src = item.videoUrl;
  }
  framePresented = false;  // 真实画面帧呈现前不绘制，避免加载态/黑帧出现
  arVideo.currentTime = 0;
  arVideo.muted = false;
  const p = arVideo.play();
  if (p) p.catch(() => {
    arVideo.muted = true;
    arVideo.play().catch(() => {});
    soundBtn.classList.add('show');
  });
  hintEl.textContent = item.title || '已识别';
  hintEl.classList.add('locked');
}

// 追踪丢失：立即停播清屏
function onLost() {
  if (state !== 'track') return;
  state = 'detect';
  targetH = null;
  framePresented = false;
  arVideo.pause();
  soundBtn.classList.remove('show');
  hintEl.textContent = '对准照片，自动识别';
  hintEl.classList.remove('locked');
  clearGL();
}

// 视频播完：复位并通知 worker 回到识别态
function onVideoEnded() {
  state = 'detect';
  targetH = null;
  framePresented = false;
  arVideo.pause();
  soundBtn.classList.remove('show');
  hintEl.textContent = '对准照片，自动识别';
  hintEl.classList.remove('locked');
  clearGL();
  if (worker) worker.postMessage({ type: 'reset' });
}

/* ---------------- worker 消息 ---------------- */
function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === 'ready') {
    workerReady = true;
    if (state === 'boot') {
      state = 'detect';
      hintEl.textContent = '对准照片，自动识别';
    }
    return;
  }
  if (m.type === 'error') {
    hintEl.textContent = '识别引擎加载失败';
    return;
  }
  inFlight = false; // done / locked / H / lost / candidates 均表示一帧处理结束
  if (m.type === 'locked') {
    const item = library[m.index];
    if (item) onLocked(item, m.H);
  } else if (m.type === 'H') {
    if (state === 'track') targetH = m.H;
  } else if (m.type === 'lost') {
    onLost();
  } else if (m.type === 'candidates') {
    for (const i of m.indexes) {
      const it = library[i];
      if (it) warmVideo(it);
    }
  }
}

/* ---------------- 事件 ---------------- */
soundBtn.addEventListener('click', () => {
  arVideo.muted = false;
  soundBtn.classList.remove('show');
});
arVideo.addEventListener('ended', onVideoEnded);
window.addEventListener('resize', resizeGL);

// 视频真实画面帧呈现标记：仅当视频处于播放状态且有新帧呈现时置位，
// 此前 GL 不绘制任何内容，加载期/黑帧期画面区域保持纯摄像头画面。
if (hasRVFC) {
  arVideo.requestVideoFrameCallback(function onVF() {
    if (!arVideo.paused && !arVideo.ended && arVideo.readyState >= 2) {
      framePresented = true;
    }
    arVideo.requestVideoFrameCallback(onVF);
  });
} else {
  arVideo.addEventListener('playing', () => { framePresented = true; });
}

/* ---------------- 屏幕常亮（AR 使用中防熄屏） ---------------- */
async function keepAwake() {
  try { await navigator.wakeLock?.request('screen'); } catch { /* 不支持则忽略 */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') keepAwake();
});

/* ---------------- 健康检查 ---------------- */
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    statusDot.classList.toggle('on', d.status === 'ok');
  } catch {
    statusDot.classList.remove('on');
  }
}

/* ---------------- 启动（摄像头授权、特征库下载、worker 内 opencv 加载三路并行） ---------------- */
async function boot() {
  hintEl.textContent = '正在加载识别引擎…';
  resizeGL();
  if (!initGL()) return;
  keepAwake();
  checkHealth();
  setInterval(checkHealth, 10000);

  worker = new Worker('cv-worker.js?v=5'); // 版本号防浏览器缓存旧 worker
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    console.error('worker 错误', e);
    hintEl.textContent = '识别引擎加载失败';
  };
  const libPromise = loadLibrary().catch((e) => {
    console.error(e);
    hintEl.textContent = '特征库加载失败';
    throw e;
  });

  await initCamera();
  try {
    await libPromise;
  } catch {
    return;
  }
  maybePreloadAll();
  schedulePump();
  requestAnimationFrame(renderLoop);
}

boot();
