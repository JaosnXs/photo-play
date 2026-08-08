/* PhotoPlay AR —— 前端全本地识别与追踪
 *
 * 流程：OpenCV.js(WASM) ORB 识别 → 光流(LK)逐帧追踪 → WebGL 透视绘制视频
 * 后端只负责静态下发：/api/library(注册表) /photos/{id} /videos/{id}
 */
'use strict';

/* ---------------- 常量 ---------------- */
const WORK_W = 800;          // 识别与追踪的统一工作宽度
const LIB_MAX_W = 1600;      // 注册图特征提取限宽
const LIB_SCALE2 = 0.5;      // 注册图第二尺度（增强尺度覆盖）
const DETECT_INTERVAL = 300; // 识别轮询 (ms)
const MIN_GOOD = 12;         // 进入几何校验的最少匹配点
const MIN_INLIERS = 10;      // 命中所需单应性内点
const TRACK_MIN_PTS = 12;    // 追踪最少保留点数
const LOST_FRAMES = 6;       // 连续多少帧追踪失败判定丢失
const LOWE_RATIO = 0.75;

/* ---------------- DOM ---------------- */
const cameraEl = document.getElementById('camera');
const glCanvas = document.getElementById('gl');
const arVideo = document.getElementById('ar-video');
const soundBtn = document.getElementById('sound-btn');
const hintEl = document.getElementById('scan-hint');
const statusDot = document.getElementById('status-dot');

/* ---------------- 状态 ---------------- */
let cvReady = false;
let orb = null;
let bf = null;
let library = [];            // {id,title,videoUrl, kps:Float32Array(归一化), des:Mat}
let state = 'boot';          // boot | detect | track
let track = null;            // {item, H:Array9, prevGray:Mat, prevPts:Array}
let lostCount = 0;
let lastDetect = 0;

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
function applyH(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}
function mat3ToGL(M) { // 行优先 -> GL 列优先
  return new Float32Array([M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]]);
}

/* ---------------- 摄像头 ---------------- */
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    cameraEl.srcObject = stream;
  } catch (e) {
    document.getElementById('fallback-tip').style.display = 'flex';
  }
}

/* ---------------- 帧采集（统一工作分辨率） ---------------- */
function grabGray() {
  const vw = cameraEl.videoWidth, vh = cameraEl.videoHeight;
  if (!vw || !vh) return null;
  const scale = Math.min(1, WORK_W / vw);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  workCanvas.width = w; workCanvas.height = h;
  workCtx.drawImage(cameraEl, 0, 0, w, h);
  const imgData = workCtx.getImageData(0, 0, w, h);
  const src = cv.matFromImageData(imgData);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  src.delete();
  return { gray, w, h };
}

/* ---------------- 特征库加载（浏览器内提取，保证一致性） ---------------- */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function extractLibFeatures(img, scale) {
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const src = cv.matFromImageData(ctx.getImageData(0, 0, w, h));
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const kps = new cv.KeyPointVector();
  const des = new cv.Mat();
  const none = new cv.Mat();
  orb.detectAndCompute(gray, none, kps, des);
  // 归一化到 [0,1] 坐标（与图像绝对尺寸解耦）
  const pts = new Float32Array(kps.size() * 2);
  for (let i = 0; i < kps.size(); i++) {
    pts[i * 2] = kps.get(i).pt.x / w;
    pts[i * 2 + 1] = kps.get(i).pt.y / h;
  }
  src.delete(); gray.delete(); none.delete(); kps.delete();
  return { pts, des };
}

async function loadLibrary() {
  const list = await (await fetch('/api/library')).json();
  const scale1 = s => s;
  for (const item of list) {
    try {
      const img = await loadImage(item.image_url);
      const s1 = Math.min(1, LIB_MAX_W / img.naturalWidth);
      // 三尺度特征拼接（1x / 0.5x / 0.25x），覆盖"大目标/中目标/小目标"
      const f1 = extractLibFeatures(img, s1);
      const f2 = extractLibFeatures(img, s1 * 0.5);
      const f3 = extractLibFeatures(img, s1 * 0.25);
      const mergedDes = new cv.Mat();
      const vec = new cv.MatVector();
      vec.push_back(f1.des); vec.push_back(f2.des); vec.push_back(f3.des);
      cv.vconcat(vec, mergedDes);
      vec.delete(); f2.des.delete(); f3.des.delete();
      const pts = new Float32Array(f1.pts.length + f2.pts.length + f3.pts.length);
      pts.set(f1.pts, 0);
      pts.set(f2.pts, f1.pts.length);
      pts.set(f3.pts, f1.pts.length + f2.pts.length);
      f1.des.delete();
      library.push({
        id: item.photo_id, title: item.title, videoUrl: item.video_url,
        kps: pts, des: mergedDes,
      });
    } catch (e) {
      console.error('加载注册图失败', item.photo_id, e);
    }
  }
}

/* ---------------- 检测（未锁定时轮询） ---------------- */
function detect(gray, workW, workH) {
  const kps = new cv.KeyPointVector();
  const qdes = new cv.Mat();
  const none = new cv.Mat();
  orb.detectAndCompute(gray, none, kps, qdes);
  if (qdes.rows === 0) { kps.delete(); qdes.delete(); none.delete(); return null; }

  const qpts = [];
  for (let i = 0; i < kps.size(); i++) qpts.push(kps.get(i).pt.x, kps.get(i).pt.y);

  let best = null;
  for (const item of library) {
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(qdes, item.des, knn, 2);
    const src = [], dst = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() === 2) {
        const m = pair.get(0), n = pair.get(1);
        // matFromArray 生成的 DMatch：第 1 维为 query
        if (m.distance < LOWE_RATIO * n.distance) {
          src.push(item.kps[m.trainIdx * 2], item.kps[m.trainIdx * 2 + 1]);   // 注册图(归一化)
          dst.push(qpts[m.queryIdx * 2], qpts[m.queryIdx * 2 + 1]);           // 当前帧(像素)
        }
      }
    }
    knn.delete();
    if (src.length / 2 < MIN_GOOD) continue;

    const srcMat = cv.matFromArray(src.length / 2, 1, cv.CV_32FC2, src);
    const dstMat = cv.matFromArray(dst.length / 2, 1, cv.CV_32FC2, dst);
    const mask = new cv.Mat();
    const Hmat = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask);
    let inliers = 0;
    const inlierLibPts = [];
    if (!Hmat.empty()) {
      for (let i = 0; i < mask.rows; i++) {
        if (mask.data[i]) {
          inliers++;
          inlierLibPts.push(src[i * 2], src[i * 2 + 1]);
        }
      }
    }
    srcMat.delete(); dstMat.delete(); mask.delete();
    if (!Hmat.empty() && inliers >= MIN_INLIERS && (!best || inliers > best.inliers)) {
      const H = Array.from(Hmat.data64F);
      best = { item, H, inliers, inlierLibPts };
    }
    Hmat.delete();
  }
  kps.delete(); qdes.delete(); none.delete();
  return best;
}

/* ---------------- 追踪（光流逐帧） ---------------- */
function startTracking(det, gray) {
  const { item, H } = det;
  // 追踪点：检测内点(注册图归一化) 映射到当前帧像素
  const pts = [];
  for (let i = 0; i < det.inlierLibPts.length; i += 2) {
    const [x, y] = applyH(H, det.inlierLibPts[i], det.inlierLibPts[i + 1]);
    pts.push(x, y);
  }
  track = { item, H, prevGray: gray.clone(), prevPts: pts };
  lostCount = 0;
  state = 'track';

  if (arVideo.dataset.photoId !== item.id) {
    arVideo.dataset.photoId = item.id;
    arVideo.src = item.videoUrl;
  }
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

function stopTracking() {
  state = 'detect';
  track = null;
  lostCount = 0;
  arVideo.pause();
  soundBtn.classList.remove('show');
  hintEl.textContent = '对准照片，自动识别';
  hintEl.classList.remove('locked');
  clearGL();
}

function trackFrame(gray) {
  if (!track) return;
  const N = track.prevPts.length / 2;
  if (N < TRACK_MIN_PTS) { if (++lostCount >= LOST_FRAMES) stopTracking(); return; }

  const prevPts = cv.matFromArray(N, 1, cv.CV_32FC2, track.prevPts);
  const nextPts = new cv.Mat();
  const status = new cv.Mat();
  const err = new cv.Mat();
  cv.calcOpticalFlowPyrLK(track.prevGray, gray, prevPts, nextPts, status, err);

  const srcArr = [], dstArr = [];
  const nextData = nextPts.data32F;
  for (let i = 0; i < N; i++) {
    if (status.data[i]) {
      srcArr.push(track.prevPts[i * 2], track.prevPts[i * 2 + 1]);
      dstArr.push(nextData[i * 2], nextData[i * 2 + 1]);
    }
  }
  prevPts.delete(); status.delete(); err.delete();

  if (srcArr.length / 2 < TRACK_MIN_PTS) {
    nextPts.delete();
    if (++lostCount >= LOST_FRAMES) stopTracking();
    return;
  }

  const srcMat = cv.matFromArray(srcArr.length / 2, 1, cv.CV_32FC2, srcArr);
  const dstMat = cv.matFromArray(dstArr.length / 2, 1, cv.CV_32FC2, dstArr);
  const mask = new cv.Mat();
  const T = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3.0, mask);

  if (T.empty()) {
    srcMat.delete(); dstMat.delete(); mask.delete(); T.delete(); nextPts.delete();
    if (++lostCount >= LOST_FRAMES) stopTracking();
    return;
  }

  // 保留单应性内点作为下一帧追踪点
  const newPts = [];
  for (let i = 0; i < mask.rows; i++) {
    if (mask.data[i]) newPts.push(dstArr[i * 2], dstArr[i * 2 + 1]);
  }

  const Tarr = Array.from(T.data64F);
  track.H = mul3(Tarr, track.H);   // H_new = T · H_old
  track.prevPts = newPts;
  track.prevGray.delete();
  track.prevGray = gray.clone();
  lostCount = 0;

  srcMat.delete(); dstMat.delete(); mask.delete(); T.delete(); nextPts.delete();
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

function drawOverlay() {
  if (!gl || !track) return;
  if (arVideo.readyState < 2) return;
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
  const M = mul3(C, mul3(S, track.H));

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

/* ---------------- 主循环 ---------------- */
function loop() {
  requestAnimationFrame(loop);
  if (!cvReady || state === 'boot') return;
  if (cameraEl.readyState < 2) return;

  const now = performance.now();
  if (state === 'detect') {
    if (now - lastDetect < DETECT_INTERVAL) return;
    lastDetect = now;
    const frame = grabGray();
    if (!frame) return;
    const det = detect(frame.gray, frame.w, frame.h);
    if (det) {
      startTracking(det, frame.gray);
    }
    frame.gray.delete();
  } else if (state === 'track' && track) {
    const frame = grabGray();
    if (!frame) return;
    trackFrame(frame.gray);
    frame.gray.delete();
    if (state === 'track') drawOverlay();
  }
}

/* ---------------- 事件 ---------------- */
soundBtn.addEventListener('click', () => {
  arVideo.muted = false;
  soundBtn.classList.remove('show');
});
arVideo.addEventListener('ended', stopTracking);
window.addEventListener('resize', resizeGL);

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    statusDot.classList.toggle('on', d.status === 'ok');
  } catch {
    statusDot.classList.remove('on');
  }
}

/* ---------------- 启动 ---------------- */
async function boot() {
  hintEl.textContent = '正在加载识别引擎…';
  resizeGL();
  if (!initGL()) return;
  await initCamera();
  checkHealth();
  setInterval(checkHealth, 10000);
  hintEl.textContent = '正在加载特征库…';
  try {
    await loadLibrary();
  } catch (e) {
    hintEl.textContent = '特征库加载失败';
    return;
  }
  hintEl.textContent = '对准照片，自动识别';
  state = 'detect';
  requestAnimationFrame(loop);
}

// OpenCV.js 加载完成后启动
function onCvReady() {
  cvReady = true;
  // 该构建未导出 ORB_create，用构造器 + setter 配置参数
  orb = new cv.ORB();
  orb.setMaxFeatures(4000);
  orb.setScaleFactor(1.2);
  orb.setNLevels(16);
  bf = new cv.BFMatcher(cv.NORM_HAMMING);
  boot();
}

if (typeof cv !== 'undefined') {
  if (typeof cv.then === 'function') {
    // 模块化构建：cv 是 Promise
    cv.then(realCv => { window.cv = realCv; onCvReady(); });
  } else if (cv.Mat) {
    onCvReady();
  } else {
    const prev = cv.onRuntimeInitialized;
    cv.onRuntimeInitialized = () => { if (prev) prev(); onCvReady(); };
  }
} else {
  hintEl.textContent = '识别引擎加载失败';
}
