import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task";

const videoInput = document.getElementById("videoInput");
videoInput.removeAttribute("capture");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const analyzeBtn = document.getElementById("analyzeBtn");
const toggleSkeletonBtn = document.getElementById("toggleSkeletonBtn");
const statusEl = document.getElementById("status");
const progress = document.getElementById("progress");
const report = document.getElementById("report");
const metricsEl = document.getElementById("metrics");

let poseLandmarker;
let frames = [];
let showSkeleton = true;
let objectUrl;
let selectedHitter = null;
let lastAcceptedHip = null;

const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

async function init() {
  try {
    statusEl.textContent = "Loading pose model…";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.70,
      minPosePresenceConfidence: 0.70,
      minTrackingConfidence: 0.75
    });
    statusEl.textContent = "Pose model ready. Choose a swing video.";
    analyzeBtn.disabled = false;
  } catch (error) {
    console.error(error);
    statusEl.textContent = "Could not load the pose model. Check your internet connection and refresh.";
  }
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  document.getElementById("stage").classList.remove("hidden");
  report.classList.add("hidden");
  frames = [];
  selectedHitter = null;
  lastAcceptedHip = null;
  statusEl.textContent = "Video ready. Tap the hitter's chest to lock onto the correct player.";
});

video.addEventListener("loadedmetadata", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

document.getElementById("stage").addEventListener("click", event => {
  if (!video.videoWidth || !video.videoHeight) return;

  const rect = video.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;

  if (x < 0 || x > 1 || y < 0 || y > 1) return;

  selectedHitter = { x, y };
  drawSelectionMarker();
  statusEl.textContent = "Hitter locked. Tap Analyze Swing.";
});

function drawSelectionMarker() {
  if (!selectedHitter || !canvas.width || !canvas.height) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.strokeStyle = "#ffd84d";
  ctx.lineWidth = Math.max(4, canvas.width * 0.004);
  ctx.beginPath();
  ctx.arc(
    selectedHitter.x * canvas.width,
    selectedHitter.y * canvas.height,
    Math.max(18, canvas.width * 0.025),
    0,
    Math.PI * 2
  );
  ctx.stroke();
  ctx.restore();
}

function resizeCanvas() {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
}

video.addEventListener("timeupdate", () => {
  if (!showSkeleton || !frames.length) return;
  drawNearestFrame(video.currentTime);
});

toggleSkeletonBtn.addEventListener("click", () => {
  showSkeleton = !showSkeleton;
  toggleSkeletonBtn.textContent = showSkeleton ? "Hide Skeleton" : "Show Skeleton";
  if (!showSkeleton) ctx.clearRect(0, 0, canvas.width, canvas.height);
  else drawNearestFrame(video.currentTime);
});

analyzeBtn.addEventListener("click", analyzeVideo);

async function analyzeVideo() {
  if (!selectedHitter) {
    statusEl.textContent = "Tap the hitter's chest first so the app analyzes the correct person.";
    return;
  }

  if (!poseLandmarker || !Number.isFinite(video.duration) || video.duration <= 0) {
    statusEl.textContent = "The video is not ready yet. Press play once, pause, then try Analyze Swing again.";
    return;
  }

  analyzeBtn.disabled = true;
  toggleSkeletonBtn.disabled = true;
  progress.classList.remove("hidden");
  progress.value = 0;
  report.classList.add("hidden");
  frames = [];
lastAcceptedHip = null;
let previousAcceptedLandmarks = null;
let identityState = {
  shoulders: false,
  hips: false
};
  const stopAt = Math.min(video.duration, 35);
  const originalMuted = video.muted;
  const originalRate = video.playbackRate;
  const originalControls = video.controls;

  statusEl.textContent = "Analyzing the video frame by frame…";

  try {
    video.pause();
    video.currentTime = 0;
    video.muted = true;
    video.playbackRate = 1;
    video.controls = false;

    await waitForVideoReady();

const sampleRate = 12;
const sampleInterval = 1 / sampleRate;
const runTimestampBase = performance.now();

for (let current = 0; current < stopAt; current += sampleInterval) {
  await seekVideoTo(current);

  const tracked = detectLockedHitter(
    Math.round(runTimestampBase + current * 1000)
  );

  if (tracked) {
const correctedPose = lockTorsoIdentity(
  tracked.landmarks,
  tracked.world,
  previousAcceptedLandmarks,
  identityState
);

identityState = correctedPose.state;

const torsoIsStable = hasStableTorso(
  correctedPose.landmarks,
  previousAcceptedLandmarks
);

if (
  isReliablePose(correctedPose.landmarks) &&
  torsoIsStable
) {
const hip = midpoint(
  correctedPose.landmarks[23],
  correctedPose.landmarks[24]
);

    const jump = lastAcceptedHip
      ? distance2D(hip, lastAcceptedHip)
      : 0;

    if (!lastAcceptedHip || jump < 0.16) {
      frames.push({
  time: current,
  landmarks: correctedPose.landmarks,
  world: correctedPose.world
});
      lastAcceptedHip = hip;

 previousAcceptedLandmarks = {
  landmarks: correctedPose.landmarks.map(point => ({ ...point })),
  world: correctedPose.world
    ? correctedPose.world.map(point => ({ ...point }))
    : null
};
    }
  }
}
  progress.value = Math.min(
    100,
    (current / stopAt) * 100
  );

  await new Promise((resolve) =>
    requestAnimationFrame(resolve)
  );
}

video.pause();
progress.value = 100;

    if (frames.length < 5) {
      throw new Error("Tracking quality was too low, so no score was produced.");
    }

    const analysis = calculateMetrics(frames);
    renderReport(analysis);
    toggleSkeletonBtn.disabled = false;
    statusEl.textContent = `Analysis complete. Pose found in ${frames.length} frames.`;

    const middleFrame = frames[Math.floor(frames.length / 2)];
    video.currentTime = middleFrame.time;
    drawNearestFrame(middleFrame.time);
  } catch (error) {
    console.error(error);
    statusEl.textContent =
      `${error.message} Use a bright open-side clip with the full body visible.`;
  } finally {
    video.pause();
    video.muted = originalMuted;
    video.playbackRate = originalRate;
    video.controls = originalControls;
    analyzeBtn.disabled = false;
    progress.classList.add("hidden");
  }
}
function seekVideoTo(time) {
  return new Promise((resolve, reject) => {
    const safeTime = Math.min(
      Math.max(0, time),
      Math.max(0, video.duration - 0.001)
    );

    if (
      video.readyState >= 2 &&
      Math.abs(video.currentTime - safeTime) < 0.002
    ) {
      requestAnimationFrame(resolve);
      return;
    }

    let completed = false;

    const cleanup = () => {
      video.removeEventListener("seeked", handleSeeked);
      video.removeEventListener("error", handleError);
      clearTimeout(timeoutId);
    };

    const handleSeeked = () => {
      if (completed) return;
      completed = true;
      cleanup();
      requestAnimationFrame(resolve);
    };

    const handleError = () => {
      if (completed) return;
      completed = true;
      cleanup();
      reject(new Error("Safari could not move through the video."));
    };

    const timeoutId = setTimeout(() => {
      if (completed) return;
      completed = true;
      cleanup();
      reject(new Error("Video frame loading timed out."));
    }, 10000);

    video.addEventListener("seeked", handleSeeked, { once: true });
    video.addEventListener("error", handleError, { once: true });

    video.currentTime = safeTime;
  });
}
function waitForVideoReady() {
  if (video.readyState >= 2) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const ready = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("Safari could not prepare this video."));
    };
    const cleanup = () => {
      video.removeEventListener("loadeddata", ready);
      video.removeEventListener("canplay", ready);
      video.removeEventListener("error", fail);
    };

    video.addEventListener("loadeddata", ready, { once: true });
    video.addEventListener("canplay", ready, { once: true });
    video.addEventListener("error", fail, { once: true });

    // Calling load helps Safari finish preparing newly selected local videos.
    video.load();
  });
}

function drawNearestFrame(time) {
  if (!frames.length) return;
  const frame = frames.reduce((best, item) =>
    Math.abs(item.time - time) < Math.abs(best.time - time) ? item : best
  );

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const drawing = new DrawingUtils(ctx);
  drawing.drawConnectors(frame.landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#45c86c",
    lineWidth: Math.max(3, canvas.width * 0.003)
  });
  drawing.drawLandmarks(frame.landmarks, {
    color: "#ffffff",
    fillColor: "#45c86c",
    radius: Math.max(3, canvas.width * 0.003)
  });
}

function detectLockedHitter(timestamp) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh || !selectedHitter) return null;

  const cropWidth = Math.min(vw, Math.round(vw * 0.62));
  const cropHeight = Math.min(vh, Math.round(vh * 0.94));

  const centerX = selectedHitter.x * vw;
  const centerY = selectedHitter.y * vh;

  const sx = Math.max(0, Math.min(vw - cropWidth, centerX - cropWidth / 2));
  const sy = Math.max(0, Math.min(vh - cropHeight, centerY - cropHeight * 0.46));

  cropCanvas.width = cropWidth;
  cropCanvas.height = cropHeight;
  cropCtx.drawImage(video, sx, sy, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

  const result = poseLandmarker.detectForVideo(cropCanvas, timestamp);
  if (!result.landmarks?.[0]) return null;

  const mapped = result.landmarks[0].map(point => ({
    ...point,
    x: (sx + point.x * cropWidth) / vw,
    y: (sy + point.y * cropHeight) / vh
  }));

  return {
    landmarks: mapped,
    world: result.worldLandmarks?.[0] || null
  };
}

function isReliablePose(points) {
  const required = [11, 12, 23, 24, 25, 26, 27, 28];
  const visibility = required.map(i => points[i]?.visibility ?? 0);
  const averageVisibility =
    visibility.reduce((sum, value) => sum + value, 0) / visibility.length;

  const xs = required.map(i => points[i].x);
  const ys = required.map(i => points[i].y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);

  const hip = midpoint(points[23], points[24]);
  const lockDistance = Math.hypot(
    hip.x - selectedHitter.x,
    hip.y - selectedHitter.y
  );

  return (
    averageVisibility >= 0.55 &&
    height >= 0.22 &&
    width >= 0.06 &&
    lockDistance <= 0.34
  );
}

function hasStableTorso(currentPoints, previousPose) {
  if (!previousPose?.landmarks) return true;

  const torsoIndexes = [11, 12, 23, 24];

  const jumps = torsoIndexes.map(index =>
    distance2D(
      currentPoints[index],
      previousPose.landmarks[index]
    )
  );

  const largestJump = Math.max(...jumps);

  const averageJump =
    jumps.reduce((sum, value) => sum + value, 0) /
    jumps.length;

  return (
    largestJump <= 0.10 &&
    averageJump <= 0.06
  );
}

function distance2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function swapPair(points, left, right) {
  if (!points) return;

  const temp = points[left];
  points[left] = points[right];
  points[right] = temp;
}
const TORSO_PAIRS = [
  { left: 11, right: 12, stateKey: "shoulders" },
  { left: 23, right: 24, stateKey: "hips" }
];

function lockTorsoIdentity(
  currentLandmarks,
  currentWorld,
  previousPose,
  currentState
) {
  const correctedLandmarks =
    currentLandmarks.map(point => ({ ...point }));

  const correctedWorld = currentWorld
    ? currentWorld.map(point => ({ ...point }))
    : null;

  const nextState = { ...currentState };

  if (!previousPose?.landmarks) {
    return {
      landmarks: correctedLandmarks,
      world: correctedWorld,
      state: nextState
    };
  }

  for (const { left, right, stateKey } of TORSO_PAIRS) {
    const currentLeft = currentLandmarks[left];
    const currentRight = currentLandmarks[right];

    const previousLeft =
      previousPose.landmarks[left];

    const previousRight =
      previousPose.landmarks[right];

    if (
      !currentLeft ||
      !currentRight ||
      !previousLeft ||
      !previousRight
    ) {
      continue;
    }

    const keepCost =
      distance2D(currentLeft, previousLeft) +
      distance2D(currentRight, previousRight);

    const swapCost =
      distance2D(currentLeft, previousRight) +
      distance2D(currentRight, previousLeft);

    /*
     * Only switch identities when the swapped assignment
     * is clearly closer to the previous accepted frame.
     */
    const shouldSwap =
      swapCost + 0.018 < keepCost;

    nextState[stateKey] = shouldSwap;

    if (shouldSwap) {
      swapPair(
        correctedLandmarks,
        left,
        right
      );

      swapPair(
        correctedWorld,
        left,
        right
      );
    }

    const correctedLeft =
      correctedLandmarks[left];

    const correctedRight =
      correctedLandmarks[right];

    const leftJump =
      distance2D(correctedLeft, previousLeft);

    const rightJump =
      distance2D(correctedRight, previousRight);

    const leftVisibility =
      correctedLeft.visibility ?? 1;

    const rightVisibility =
      correctedRight.visibility ?? 1;

    /*
     * When a shoulder or hip becomes hidden and suddenly
     * jumps, hold its previous reliable location.
     */
    if (
      leftJump > 0.075 &&
      leftVisibility < 0.72
    ) {
      correctedLandmarks[left] = {
        ...previousLeft
      };

      if (
        correctedWorld &&
        previousPose.world?.[left]
      ) {
        correctedWorld[left] = {
          ...previousPose.world[left]
        };
      }
    }

    if (
      rightJump > 0.075 &&
      rightVisibility < 0.72
    ) {
      correctedLandmarks[right] = {
        ...previousRight
      };

      if (
        correctedWorld &&
        previousPose.world?.[right]
      ) {
        correctedWorld[right] = {
          ...previousPose.world[right]
        };
      }
    }
  }

  return {
    landmarks: correctedLandmarks,
    world: correctedWorld,
    state: nextState
  };
}
function midpoint(a, b) {
  return { x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:((a.z||0)+(b.z||0))/2 };
}

function distance(a, b) {
  return Math.hypot(a.x-b.x, a.y-b.y, (a.z||0)-(b.z||0));
}

function angle(a, b, c) {
  const ab = {x:a.x-b.x, y:a.y-b.y, z:(a.z||0)-(b.z||0)};
  const cb = {x:c.x-b.x, y:c.y-b.y, z:(c.z||0)-(b.z||0)};
  const dot = ab.x*cb.x + ab.y*cb.y + ab.z*cb.z;
  const mag = Math.hypot(ab.x,ab.y,ab.z) * Math.hypot(cb.x,cb.y,cb.z);
  return Math.acos(Math.max(-1,Math.min(1,dot/(mag||1)))) * 180/Math.PI;
}

function orientation(a, b) {
  return Math.atan2(b.z-a.z, b.x-a.x) * 180/Math.PI;
}

function wrapAngle(value) {
  let v = Math.abs(value);
  while (v > 180) v = Math.abs(v - 360);
  return v;
}

function percentileRange(values) {
  const sorted = [...values].sort((a,b)=>a-b);
  const lo = sorted[Math.floor(sorted.length*0.1)];
  const hi = sorted[Math.floor(sorted.length*0.9)];
  return hi-lo;
}

function clamp(n,min,max){ return Math.max(min,Math.min(max,n)); }
function scoreNear(value,target,tolerance){
  return Math.round(clamp(100 - Math.abs(value-target)/tolerance*45, 0, 100));
}
function scoreBelow(value,good,bad){
  return Math.round(clamp(100 - ((value-good)/(bad-good))*100,0,100));
}

function calculateFrontLegStability(items) {
  if (items.length < 5) {
    return {
      kneeAngle: 0,
      firmingChange: 0,
      score: 0
    };
  }

  const wristSpeeds = items.map((frame, index) => {
    if (index === 0) {
      return 0;
    }

    const previousPoints =
      items[index - 1].world ||
      items[index - 1].landmarks;

    const currentPoints =
      frame.world ||
      frame.landmarks;

    const previousWrists = midpoint(
      previousPoints[15],
      previousPoints[16]
    );

    const currentWrists = midpoint(
      currentPoints[15],
      currentPoints[16]
    );

    const timeDifference =
      frame.time - items[index - 1].time || 1;

    return (
      distance(currentWrists, previousWrists) /
      timeDifference
    );
  });

  const contactIndex =
    wristSpeeds.indexOf(
      Math.max(...wristSpeeds)
    );

  const contactPoints =
    items[contactIndex].world ||
    items[contactIndex].landmarks;

  const leftKneeAngle = angle(
    contactPoints[23],
    contactPoints[25],
    contactPoints[27]
  );

  const rightKneeAngle = angle(
    contactPoints[24],
    contactPoints[26],
    contactPoints[28]
  );

  /*
   * The lead leg usually appears firmer than the rear leg
   * near contact. This lets the app estimate the lead side
   * without asking whether the hitter is right- or left-handed.
   */
  const leadIndexes =
    leftKneeAngle >= rightKneeAngle
      ? {
          hip: 23,
          knee: 25,
          ankle: 27
        }
      : {
          hip: 24,
          knee: 26,
          ankle: 28
        };

  const beforeContactIndex =
    Math.max(
      0,
      contactIndex - 3
    );

  const beforePoints =
    items[beforeContactIndex].world ||
    items[beforeContactIndex].landmarks;

  const kneeAngleBeforeContact = angle(
    beforePoints[leadIndexes.hip],
    beforePoints[leadIndexes.knee],
    beforePoints[leadIndexes.ankle]
  );

  const kneeAngleAtContact = angle(
    contactPoints[leadIndexes.hip],
    contactPoints[leadIndexes.knee],
    contactPoints[leadIndexes.ankle]
  );

  const firmingChange =
    kneeAngleAtContact -
    kneeAngleBeforeContact;

  const straightnessScore =
    clamp(
      ((kneeAngleAtContact - 120) / 40) * 100,
      0,
      100
    );

  const firmingScore =
    clamp(
      ((firmingChange + 5) / 20) * 100,
      0,
      100
    );

  const score =
    Math.round(
      (
        straightnessScore +
        firmingScore
      ) / 2
    );

  return {
    kneeAngle:
      Math.round(kneeAngleAtContact),

    firmingChange:
      Math.round(firmingChange),

    score
  };
}

function trimTrailingStillFrames(items) {
  if (items.length < 8) return items;

  const wristSpeeds = [];

  for (let i = 1; i < items.length; i++) {
    const previousPoints =
      items[i - 1].world || items[i - 1].landmarks;

    const currentPoints =
      items[i].world || items[i].landmarks;

    const previousWrists = midpoint(
      previousPoints[15],
      previousPoints[16]
    );

    const currentWrists = midpoint(
      currentPoints[15],
      currentPoints[16]
    );

    const timeDifference =
      items[i].time - items[i - 1].time || 1;

    wristSpeeds.push({
      frameIndex: i,
      speed:
        distance(currentWrists, previousWrists) /
        timeDifference
    });
  }

  const peakSpeed = Math.max(
    ...wristSpeeds.map(item => item.speed)
  );

  if (!Number.isFinite(peakSpeed) || peakSpeed <= 0) {
    return items;
  }

  const movementThreshold = peakSpeed * 0.12;

  const movingFrames = wristSpeeds.filter(
    item => item.speed >= movementThreshold
  );

  if (!movingFrames.length) return items;

  const lastMovingFrame =
    movingFrames[movingFrames.length - 1].frameIndex;

  const finishBuffer = 3;

  const endingFrame = Math.min(
    items.length,
    lastMovingFrame + finishBuffer + 1
  );

  if (endingFrame < 5) return items;

  return items.slice(0, endingFrame);
}

function calculateMetrics(items) {
  const activeItems = trimTrailingStillFrames(items);

  const valid = activeItems.filter(f => f.world);
  const source =
    valid.length >= 5 ? valid : activeItems;

  const kneeAngles = [];
  const hipAngles = [];
  const shoulderAngles = [];
  const hipCenters = [];
  const shoulderCenters = [];
  const bodyScales = [];

  for (const f of source) {
    const p = f.world || f.landmarks;
    const hip = midpoint(p[23], p[24]);
    const shoulder = midpoint(p[11], p[12]);
    hipCenters.push(hip);
    shoulderCenters.push(shoulder);
    bodyScales.push(distance(hip, shoulder) + distance(shoulder, midpoint(p[7],p[8])));
    kneeAngles.push(Math.min(angle(p[23],p[25],p[27]), angle(p[24],p[26],p[28])));
    hipAngles.push(orientation(p[23],p[24]));
    shoulderAngles.push(orientation(p[11],p[12]));
  }

  const bodyScale = bodyScales.reduce((a,b)=>a+b,0)/bodyScales.length || 1;
  const kneeBend = Math.round(kneeAngles.reduce((a,b)=>a+b,0)/kneeAngles.length);
  const hipRotation = Math.round(wrapAngle(Math.max(...hipAngles)-Math.min(...hipAngles)));
  const shoulderRotation = Math.round(wrapAngle(Math.max(...shoulderAngles)-Math.min(...shoulderAngles)));
    const horizontalHip =
    percentileRange(
      hipCenters.map(point => point.x)
    ) / bodyScale;

  const verticalHip =
    percentileRange(
      hipCenters.map(point => point.y)
    ) / bodyScale;

  const separationAngles =
    hipAngles.map(
      (hipAngle, index) =>
        wrapAngle(
          hipAngle - shoulderAngles[index]
        )
    );

  const maxSeparation =
    Math.round(
      Math.max(...separationAngles)
    );

    const timeToContact =
    estimateTimeToContact(source);

  const frontLeg =
    calculateFrontLegStability(source);

  const metricScores = {
    knee: scoreNear(kneeBend, 105, 45),

        frontLeg:
      frontLeg.score,

    horizontal: scoreBelow(
      horizontalHip,
      0.10,
      0.55
    ),

    vertical: scoreBelow(
      verticalHip,
      0.08,
      0.40
    ),

    separation: Math.round(
      clamp(
        (maxSeparation / 25) * 100,
        0,
        100
      )
    ),

    timing: scoreBelow(
      timeToContact,
      0.22,
      0.65
    )
  };

  const overall = Math.round(Object.values(metricScores).reduce((a,b)=>a+b,0)/6);

  const issues = [
    {key:"horizontal", score:metricScores.horizontal, title:"Excessive hip slide",
      why:"Too much forward hip travel can reduce balance and make clean rotation harder.",
      one:"Rotate around a stable center instead of carrying the hips toward the pitcher.",
      drill:"Pause-at-Land Drill — 3 rounds of 5 controlled swings."},
    {key:"vertical", score:metricScores.vertical, title:"Too much vertical hip movement",
      why:"Large up-and-down movement can redirect energy away from efficient rotation.",
      one:"Keep the belt line level from load through contact.",
      drill:"Chair-Height Tee Drill — 3 rounds of 5 swings."},
       {
      key: "frontLeg",
      score: metricScores.frontLeg,
      title: "Front leg needs to firm up",

      why:
        "The lead knee did not appear to create a firm base as the hitter approached contact.",

      one:
        "Land under control, then allow the front leg to firm while the back hip comes through.",

      drill:
        "Front-Leg Brace Drill — 3 rounds of 5 controlled swings."
    },
        {
      key: "separation",
      score: metricScores.separation,
      title: "Limited hip-shoulder separation",

      why:
        "The hips and shoulders appeared to rotate together instead of creating stretch through the torso.",

      one:
        "Allow the hips to begin turning while keeping the chest closed for a moment longer.",

      drill:
        "Separation Drill — 3 rounds of 5 deliberate swings."
    },
    {key:"knee", score:metricScores.knee, title:"Back-knee position",
      why:"Knee position affects posture, stability, and the hitter’s ability to rotate.",
      one:"Maintain athletic knee flex without collapsing.",
      drill:"Hold-the-Finish Tee Drill — 3 rounds of 5 swings."},
    {key:"timing", score:metricScores.timing, title:"Long move to contact",
      why:"A longer movement window can make late adjustment more difficult.",
      one:"Create a quiet load and a direct turn to the ball.",
      drill:"Short Toss Decision Drill — 3 rounds of 5 swings."}
  ].sort((a,b)=>a.score-b.score);

  return {
    overall,
    metricScores,
    metrics:[
  [
    "Knee Bend",
    `${kneeBend}°`,
    metricScores.knee,
    "Athletic posture throughout the swing"
  ],

  [
    "Front-Leg Stability",
    `${frontLeg.kneeAngle}°`,
    metricScores.frontLeg,
    `${
      frontLeg.firmingChange >= 0
        ? "+"
        : ""
    }${frontLeg.firmingChange}° firming approaching contact`
  ]
],
    issue:issues[0]
  };
}

function estimateTimeToContact(items) {
  const wrists = items.map(f => {
    const p = f.world || f.landmarks;
    const w = midpoint(p[15],p[16]);
    return {time:f.time, speed:0, point:w};
  });
  for(let i=1;i<wrists.length;i++){
    const dt=wrists[i].time-wrists[i-1].time||1;
    wrists[i].speed=distance(wrists[i].point,wrists[i-1].point)/dt;
  }
  const peak=Math.max(...wrists.map(w=>w.speed));
  const threshold=peak*0.18;
  const start=wrists.find(w=>w.speed>threshold)?.time ?? wrists[0].time;
  const contact=wrists.find(w=>w.speed>peak*0.82)?.time ?? wrists[wrists.length-1].time;
  return clamp(contact-start,0.05,1.5);
}

function renderReport(a) {
  report.classList.remove("hidden");
  document.getElementById("overallScore").textContent = a.overall;
  document.querySelector(".scoreRing").style.background =
    `conic-gradient(${a.overall>=75?"#45c86c":a.overall>=55?"#ff9827":"#ff4545"} ${a.overall*3.6}deg,#35434d 0deg)`;
  document.getElementById("headline").textContent =
    `Primary Focus: ${a.issue.title}`;
   document.getElementById("summary").textContent =
    `The swing was analyzed using ${frames.length} accepted tracking frames. Your strongest opportunity for improvement is "${a.issue.title}".`;

   const lowestScore =
    Math.min(
      ...a.metrics.map(
        metric => metric[2]
      )
    );

  metricsEl.innerHTML =
    a.metrics.map(
      ([name, value, score, note]) => {
        const rating =
          score >= 85
            ? "Excellent"
            : score >= 75
              ? "Strong"
              : score >= 55
                ? "Developing"
                : "Needs Work";

        const isPrimaryFocus =
          score === lowestScore;

        return `
          <div class="metric ${
            score >= 75
              ? "good"
              : score >= 55
                ? "warn"
                : "bad"
          }">
            <h3>${name}</h3>

            <strong>${value}</strong>

            <small>
              ${rating} • Score ${score}/100
              <br>
              ${note}
              ${
                isPrimaryFocus
                  ? "<br><strong>← Work Here First</strong>"
                  : ""
              }
            </small>
          </div>
        `;
      }
    ).join("");

  document.getElementById(
    "primaryIssue"
  ).textContent =
    a.issue.title;

  document.getElementById(
    "why"
  ).textContent =
    a.issue.why;

  document.getElementById(
    "oneThing"
  ).textContent =
    a.issue.one;

  document.getElementById(
    "drill"
  ).textContent =
    a.issue.drill;
  report.scrollIntoView({behavior:"smooth"});
}

init();
