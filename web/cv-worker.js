/* PhotoPlay AR —— 识别与追踪 Worker
 *
 * 主线程只负责摄像头采集与 WebGL 渲染；
 * ORB 识别（合并索引单次匹配）与 LK 光流追踪全部在此 Worker 线程执行，
 * CV 计算不再阻塞渲染，视频叠加帧率不受识别耗时影响。
 * 注意：对单应性不做任何滤波，渲染帧直接使用追踪解算结果，
 * 保证视频"粘贴"在照片上的跟随手感。
 *
 * 消息协议（主 -> Worker）：
 *   {type:'library', items:[{n, ptsBuf, desBuf}]}  特征库（Transferable 零拷贝）
 *   {type:'frame', buf, w, h}                      RGBA 帧（Transferable）
 *   {type:'reset'}                                 视频播完，重置回识别态
 * 消息协议（Worker -> 主）：
 *   {type:'ready'}                                 cv 与特征库就绪
 *   {type:'locked', index, H}                      识别命中并进入追踪
 *   {type:'H', H}                                  追踪中的最新单应性
 *   {type:'lost'}                                  追踪丢失
 *   {type:'candidates', indexes}                   疑似候选（供视频预热）
 *   {type:'done'}                                  一帧处理完毕（流控）
 */
'use strict';

importScripts('vendor/opencv.js');

/* ---------------- 常量 ---------------- */
const QUERY_FEATURES = 1000; // 查询帧 ORB 特征上限（匹配耗时与之成正比）
const QUERY_N_LEVELS = 8;    // 查询帧金字塔层数（库端多尺度已覆盖尺度差，取半提速）
const MIN_GOOD = 12;         // 进入几何校验的最少匹配点
const MIN_INLIERS = 10;      // 命中所需单应性内点
const CANDIDATE_GOOD = 6;    // 某图匹配数达此值即上报候选（供视频预热）
const TRACK_MIN_PTS = 12;    // 追踪最少保留点数
const LOST_FRAMES = 6;       // 连续多少帧追踪失败判定丢失
const LOWE_RATIO = 0.75;
const DETECT_INTERVAL = 300; // 识别轮询 (ms)

/* ---------------- 状态 ---------------- */
let cvReady = false;
let orb = null;
let bf = null;
let library = [];       // {kps:Float32Array(归一化), offset}
let mergedDes = null;   // 全库合并描述子
let descOwner = null;   // Int32Array: 合并描述子行号 -> library 下标
let state = 'detect';   // detect | track
let track = null;       // {H:Array9, prevGray:Mat, prevPts:Array}
let lostCount = 0;
let lastDetect = 0;
let rawLib = null;      // cv 就绪前收到的特征库

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

/* ---------------- 特征库索引构建 ---------------- */
function buildIndex(items) {
  library = [];
  const desList = [];
  const owners = [];
  let total = 0;
  for (const it of items) {
    const des = cv.matFromArray(it.n, 32, cv.CV_8U, new Uint8Array(it.desBuf));
    library.push({ kps: new Float32Array(it.ptsBuf), offset: total });
    desList.push(des);
    for (let r = 0; r < it.n; r++) owners.push(library.length - 1);
    total += it.n;
  }
  if (desList.length) {
    const vec = new cv.MatVector();
    for (const d of desList) vec.push_back(d);
    mergedDes = new cv.Mat();
    cv.vconcat(vec, mergedDes);
    vec.delete();
    for (const d of desList) d.delete();
    descOwner = new Int32Array(owners);
  }
}

function tryReady() {
  if (cvReady && rawLib) {
    buildIndex(rawLib);
    rawLib = null;
    self.postMessage({ type: 'ready' });
  }
}

/* ---------------- 检测（合并索引一次匹配） ---------------- */
function detect(gray) {
  const kps = new cv.KeyPointVector();
  const qdes = new cv.Mat();
  const none = new cv.Mat();
  orb.detectAndCompute(gray, none, kps, qdes);
  if (qdes.rows === 0) { kps.delete(); qdes.delete(); none.delete(); return null; }

  const qpts = [];
  for (let i = 0; i < kps.size(); i++) qpts.push(kps.get(i).pt.x, kps.get(i).pt.y);
  kps.delete();

  // 一次 knnMatch 匹配整个合并索引，按归属照片分组收集匹配对
  const knn = new cv.DMatchVectorVector();
  bf.knnMatch(qdes, mergedDes, knn, 2);
  const srcByItem = library.map(() => []);
  const dstByItem = library.map(() => []);
  for (let i = 0; i < knn.size(); i++) {
    const pair = knn.get(i);
    if (pair.size() === 2) {
      const m = pair.get(0), n = pair.get(1);
      if (m.distance < LOWE_RATIO * n.distance) {
        const li = descOwner[m.trainIdx];
        const local = m.trainIdx - library[li].offset;
        srcByItem[li].push(library[li].kps[local * 2], library[li].kps[local * 2 + 1]); // 注册图(归一化)
        dstByItem[li].push(qpts[m.queryIdx * 2], qpts[m.queryIdx * 2 + 1]);             // 当前帧(像素)
      }
    }
  }
  knn.delete(); qdes.delete(); none.delete();

  // 仅对有效匹配数达标的候选照片做单应性几何校验，取内点最多者
  let best = null;
  const candidates = [];
  for (let li = 0; li < library.length; li++) {
    const src = srcByItem[li], dst = dstByItem[li];
    const good = src.length / 2;
    if (good >= CANDIDATE_GOOD) candidates.push(li);
    if (good < MIN_GOOD) continue;

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
      best = { li, H: Array.from(Hmat.data64F), inliers, inlierLibPts };
    }
    Hmat.delete();
  }

  if (candidates.length) self.postMessage({ type: 'candidates', indexes: candidates });
  return best;
}

/* ---------------- 追踪（光流逐帧） ---------------- */
function startTrack(det, gray) {
  // 追踪点：检测内点(注册图归一化) 映射到当前帧像素
  const pts = [];
  for (let i = 0; i < det.inlierLibPts.length; i += 2) {
    const [x, y] = applyH(det.H, det.inlierLibPts[i], det.inlierLibPts[i + 1]);
    pts.push(x, y);
  }
  track = { H: det.H, prevGray: gray.clone(), prevPts: pts };
  lostCount = 0;
  state = 'track';
  self.postMessage({ type: 'locked', index: det.li, H: det.H });
}

function registerLoss() {
  if (++lostCount >= LOST_FRAMES) {
    if (track) { track.prevGray.delete(); track = null; }
    state = 'detect';
    lostCount = 0;
    self.postMessage({ type: 'lost' });
  }
}

function trackFrame(gray) {
  if (!track) { state = 'detect'; return; }
  const N = track.prevPts.length / 2;
  if (N < TRACK_MIN_PTS) { registerLoss(); return; }

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
    registerLoss();
    return;
  }

  const srcMat = cv.matFromArray(srcArr.length / 2, 1, cv.CV_32FC2, srcArr);
  const dstMat = cv.matFromArray(dstArr.length / 2, 1, cv.CV_32FC2, dstArr);
  const mask = new cv.Mat();
  const T = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3.0, mask);

  if (T.empty()) {
    srcMat.delete(); dstMat.delete(); mask.delete(); T.delete(); nextPts.delete();
    registerLoss();
    return;
  }

  // 保留单应性内点作为下一帧追踪点
  const newPts = [];
  for (let i = 0; i < mask.rows; i++) {
    if (mask.data[i]) newPts.push(dstArr[i * 2], dstArr[i * 2 + 1]);
  }

  const Tarr = Array.from(T.data64F);
  track.H = mul3(Tarr, track.H);   // H_new = T · H_old
  const hs = track.H[8] || 1;      // 归一化，防止累乘尺度漂移
  for (let i = 0; i < 9; i++) track.H[i] /= hs;
  track.prevPts = newPts;
  track.prevGray.delete();
  track.prevGray = gray.clone();
  lostCount = 0;

  srcMat.delete(); dstMat.delete(); mask.delete(); T.delete(); nextPts.delete();
  self.postMessage({ type: 'H', H: track.H });
}

/* ---------------- 帧入口 ---------------- */
function onFrame(buf, w, h) {
  if (!cvReady || !mergedDes) { self.postMessage({ type: 'done' }); return; }
  if (state === 'detect') {
    const now = performance.now();
    if (now - lastDetect < DETECT_INTERVAL) {
      self.postMessage({ type: 'done' });
      return;
    }
    lastDetect = now;
  }
  const src = cv.matFromArray(h, w, cv.CV_8UC4, new Uint8Array(buf));
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  src.delete();

  if (state === 'detect') {
    const det = detect(gray);
    if (det) startTrack(det, gray);
  } else {
    trackFrame(gray);
  }
  gray.delete();
  self.postMessage({ type: 'done' });
}

/* ---------------- 消息入口 ---------------- */
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'library') {
    rawLib = m.items;
    tryReady();
  } else if (m.type === 'frame') {
    onFrame(m.buf, m.w, m.h);
  } else if (m.type === 'reset') {
    if (track) { track.prevGray.delete(); track = null; }
    state = 'detect';
    lostCount = 0;
  }
};

/* ---------------- OpenCV 初始化 ---------------- */
function onCvReady() {
  orb = new cv.ORB();
  orb.setMaxFeatures(QUERY_FEATURES);
  orb.setScaleFactor(1.2);
  orb.setNLevels(QUERY_N_LEVELS);
  bf = new cv.BFMatcher(cv.NORM_HAMMING);
  cvReady = true;
  tryReady();
}

if (typeof cv !== 'undefined') {
  if (typeof cv.then === 'function') {
    // 模块化构建：cv 是 Promise
    cv.then(realCv => { cv = realCv; onCvReady(); });
  } else if (cv.Mat) {
    onCvReady();
  } else {
    const prev = cv.onRuntimeInitialized;
    cv.onRuntimeInitialized = () => { if (prev) prev(); onCvReady(); };
  }
} else {
  self.postMessage({ type: 'error', message: 'opencv.js 加载失败' });
}
