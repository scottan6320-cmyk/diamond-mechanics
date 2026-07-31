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
function detectSwingWindow(items) {
  const finalIndex =
    Math.max(
      0,
      items.length - 1
    );

  if (items.length < 5) {
    return {
      launchIndex: 0,
      heelPlantIndex: 0,
      contactIndex: finalIndex,
      finishIndex: finalIndex,
      confidence: 0,
      heelPlantConfidence: 0,
      leadAnkleIndex: null,
      frames: items
    };
  }

  /*
   * HAND SPEED
   *
   * This remains our temporary contact estimate.
   */
  const wristSpeeds =
    items.map(
      (frame, index) => {
        if (index === 0) {
          return 0;
        }

        const previousPoints =
          items[index - 1].landmarks;

        const currentPoints =
          frame.landmarks;

        const previousWrists =
          midpoint(
            previousPoints[15],
            previousPoints[16]
          );

        const currentWrists =
          midpoint(
            currentPoints[15],
            currentPoints[16]
          );

        const timeDifference =
          frame.time -
          items[index - 1].time;

        if (
          !Number.isFinite(
            timeDifference
          ) ||
          timeDifference <= 0
        ) {
          return 0;
        }

        return (
          distance2D(
            currentWrists,
            previousWrists
          ) /
          timeDifference
        );
      }
    );

  const peakHandSpeed =
    Math.max(
      ...wristSpeeds
    );

  if (
    !Number.isFinite(
      peakHandSpeed
    ) ||
    peakHandSpeed <= 0
  ) {
    return {
      launchIndex: 0,
      heelPlantIndex: 0,
      contactIndex: finalIndex,
      finishIndex: finalIndex,
      confidence: 0,
      heelPlantConfidence: 0,
      leadAnkleIndex: null,
      frames: items
    };
  }

  const contactIndex =
    wristSpeeds.indexOf(
      peakHandSpeed
    );

  /*
   * TEMPORARY LAUNCH ESTIMATE
   *
   * Launch begins shortly before clear hand acceleration.
   */
  const launchThreshold =
    peakHandSpeed * 0.18;

  let firstMovingIndex = -1;

  for (
    let index = 1;
    index < contactIndex;
    index++
  ) {
    const currentSpeed =
      wristSpeeds[index];

    const nextSpeed =
      wristSpeeds[index + 1] ?? 0;

    if (
      currentSpeed >=
        launchThreshold &&
      nextSpeed >=
        launchThreshold
    ) {
      firstMovingIndex = index;
      break;
    }
  }

  const launchIndex =
    firstMovingIndex >= 0
      ? Math.max(
          0,
          firstMovingIndex - 2
        )
      : Math.max(
          0,
          contactIndex - 5
        );

  /*
   * IDENTIFY THE STRIDE / LEAD ANKLE
   *
   * The stride foot should show more screen movement
   * between the early stance frames and estimated contact.
   */
  const baselineFrameCount =
    Math.min(
      4,
      Math.max(
        1,
        contactIndex
      )
    );

  const averagePoint = (
    landmarkIndex
  ) => {
    const points =
      items
        .slice(
          0,
          baselineFrameCount
        )
        .map(
          frame =>
            frame.landmarks[
              landmarkIndex
            ]
        )
        .filter(Boolean);

    if (!points.length) {
      return null;
    }

    return {
      x:
        points.reduce(
          (sum, point) =>
            sum + point.x,
          0
        ) / points.length,

      y:
        points.reduce(
          (sum, point) =>
            sum + point.y,
          0
        ) / points.length
    };
  };

  const leftBaseline =
    averagePoint(27);

  const rightBaseline =
    averagePoint(28);

  const contactPoints =
    items[contactIndex]
      .landmarks;

  const leftTravel =
    leftBaseline &&
    contactPoints[27]
      ? distance2D(
          leftBaseline,
          contactPoints[27]
        )
      : 0;

  const rightTravel =
    rightBaseline &&
    contactPoints[28]
      ? distance2D(
          rightBaseline,
          contactPoints[28]
        )
      : 0;

  const leadAnkleIndex =
    leftTravel >= rightTravel
      ? 27
      : 28;

  /*
   * STRIDE-FOOT SPEED
   *
   * Heel plant is estimated when the stride foot has clearly
   * moved, then rapidly settles before estimated contact.
   */
  const ankleSpeeds =
    items.map(
      (frame, index) => {
        if (index === 0) {
          return 0;
        }

        const previousAnkle =
          items[index - 1]
            .landmarks[
              leadAnkleIndex
            ];

        const currentAnkle =
          frame.landmarks[
            leadAnkleIndex
          ];

        if (
          !previousAnkle ||
          !currentAnkle
        ) {
          return 0;
        }

        const timeDifference =
          frame.time -
          items[index - 1].time;

        if (
          !Number.isFinite(
            timeDifference
          ) ||
          timeDifference <= 0
        ) {
          return 0;
        }

        return (
          distance2D(
            currentAnkle,
            previousAnkle
          ) /
          timeDifference
        );
      }
    );

  const strideSearchEnd =
    Math.max(
      2,
      contactIndex
    );

  const peakStrideSpeed =
    Math.max(
      ...ankleSpeeds.slice(
        0,
        strideSearchEnd + 1
      )
    );

  const movingThreshold =
    peakStrideSpeed * 0.28;

  const settledThreshold =
    peakStrideSpeed * 0.12;

  let strideWasMoving = false;
  let heelPlantIndex = -1;

  for (
    let index = 1;
    index < contactIndex;
    index++
  ) {
    const speed =
      ankleSpeeds[index];

    const nextSpeed =
      ankleSpeeds[index + 1] ??
      speed;

    if (
      speed >= movingThreshold
    ) {
      strideWasMoving = true;
    }

    /*
     * Require the foot to settle for consecutive frames.
     * This prevents one noisy low-speed frame from being
     * labeled as heel plant.
     */
    if (
      strideWasMoving &&
      speed <= settledThreshold &&
      nextSpeed <= settledThreshold
    ) {
      heelPlantIndex = index;
      break;
    }
  }

  /*
   * Safe fallback:
   * If foot settling was not detected, estimate heel plant
   * shortly before contact rather than using the entire load.
   */
  if (heelPlantIndex < 0) {
    heelPlantIndex =
      Math.max(
        launchIndex,
        contactIndex - 4
      );
  }

  /*
   * Heel plant must occur before contact.
   */
  heelPlantIndex =
    clamp(
      heelPlantIndex,
      launchIndex,
      Math.max(
        launchIndex,
        contactIndex - 1
      )
    );

  const finishIndex =
    Math.min(
      finalIndex,
      contactIndex + 3
    );

  const swingFrameCount =
    contactIndex -
    launchIndex +
    1;

  const confidence =
    swingFrameCount >= 4 &&
    contactIndex > launchIndex
      ? 100
      : 50;

  const heelPlantConfidence =
    peakStrideSpeed > 0 &&
    strideWasMoving
      ? 85
      : 45;

  return {
    launchIndex,
    heelPlantIndex,
    contactIndex,
    finishIndex,
    confidence,
    heelPlantConfidence,
    leadAnkleIndex,

    frames:
      items.slice(
        launchIndex,
        finishIndex + 1
      )
  };
}
function calculateFrontLegStability(items, swingWindow) {
  if (items.length < 5) {
    return {
      kneeAngle: 0,
      firmingChange: 0,
      score: 0,
      confidence: 0
    };
  }


  const contactIndex =
    swingWindow.contactIndex;

  if (
    contactIndex < 0 ||
    contactIndex >= items.length
  ) {
    return {
      kneeAngle: 0,
      firmingChange: 0,
      score: 0,
      confidence: 0
    };
  }

  const contactPoints =
    items[contactIndex].world ||
    items[contactIndex].landmarks;

  if (!contactPoints) {
    return {
      kneeAngle: 0,
      firmingChange: 0,
      score: 0,
      confidence: 0
    };
  }

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
   * near contact. This estimates the lead side without
   * asking whether the hitter is right- or left-handed.
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

  /*
   * Compare the lead knee shortly after launch with the
   * same knee at the estimated contact frame.
   */
    /*
   * Compare the lead leg at heel plant with the
   * same leg near estimated contact.
   */
  const beforeContactIndex =
    swingWindow.heelPlantIndex;

  const beforePoints =
    items[beforeContactIndex].world ||
    items[beforeContactIndex].landmarks;

  if (!beforePoints) {
    return {
      kneeAngle: 0,
      firmingChange: 0,
      score: 0,
      confidence: 0
    };
  }

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
      (
        (kneeAngleAtContact - 120) /
        40
      ) * 100,
      0,
      100
    );

  const firmingScore =
    clamp(
      (
        (firmingChange + 5) /
        20
      ) * 100,
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
      Math.round(
        kneeAngleAtContact
      ),

    firmingChange:
      Math.round(
        firmingChange
      ),

    score,

    confidence:
      swingWindow.confidence
  };
}

function calculateHeadStability(items, swingWindow) {
  if (items.length < 5) {
    return {
      horizontalMovement: 0,
      verticalMovement: 0,
      score: 0,
      confidence: 0
    };
  }

      /*
   * Movement during the load is allowed.
   * Head stability is evaluated after heel plant,
   * when the hitter begins delivering the swing.
   */
  const swingFrames =
    items.slice(
      swingWindow.heelPlantIndex,
      swingWindow.contactIndex + 1
    );

  if (swingFrames.length < 3) {
    return {
      horizontalMovement: 0,
      verticalMovement: 0,
      score: 0,
      confidence: 0
    };
  }

  const horizontalOffsets = [];
  const verticalOffsets = [];

  for (const frame of swingFrames) {
    const points =
      frame.landmarks;

    const requiredIndexes = [
      0,
      7,
      8,
      11,
      12,
      23,
      24
    ];

    const hasRequiredPoints =
      requiredIndexes.every(
        index =>
          points[index] &&
          Number.isFinite(
            points[index].x
          ) &&
          Number.isFinite(
            points[index].y
          )
      );

    if (!hasRequiredPoints) {
      continue;
    }

    const hipCenter = midpoint(
      points[23],
      points[24]
    );

    const shoulderCenter = midpoint(
      points[11],
      points[12]
    );

    const earCenter = midpoint(
      points[7],
      points[8]
    );

    /*
     * Combining the nose and both ears reduces movement
     * caused by one facial landmark briefly drifting.
     */
    const headCenter = midpoint(
      points[0],
      earCenter
    );

    const torsoHeight =
      distance2D(
        shoulderCenter,
        hipCenter
      );

    if (
      !Number.isFinite(torsoHeight) ||
      torsoHeight <= 0.001
    ) {
      continue;
    }

    /*
     * Measure the head relative to the hips rather than
     * relative to the camera. This helps prevent ordinary
     * body movement from being mistaken for head instability.
     */
    horizontalOffsets.push(
      (
        headCenter.x -
        hipCenter.x
      ) / torsoHeight
    );

    verticalOffsets.push(
      (
        headCenter.y -
        hipCenter.y
      ) / torsoHeight
    );
  }

  if (
    horizontalOffsets.length < 3 ||
    verticalOffsets.length < 3
  ) {
    return {
      horizontalMovement: 0,
      verticalMovement: 0,
      score: 0,
      confidence: 0
    };
  }

  /*
   * Percentile range ignores an occasional tracking spike
   * while preserving the hitter's normal movement pattern.
   */
  const horizontalMovement =
    percentileRange(
      horizontalOffsets
    );

  const verticalMovement =
    percentileRange(
      verticalOffsets
    );

  const horizontalScore =
    scoreBelow(
      horizontalMovement,
      0.08,
      0.35
    );

  const verticalScore =
    scoreBelow(
      verticalMovement,
      0.08,
      0.30
    );

  const score =
    Math.round(
      horizontalScore * 0.7 +
      verticalScore * 0.3
    );

  /*
   * Confidence currently describes whether we found a
   * believable swing window and enough usable head frames.
   * It does not describe swing quality.
   */
  const usableFrameRate =
    horizontalOffsets.length /
    swingFrames.length;

  const confidence =
    Math.round(
      clamp(
        swingWindow.confidence *
        usableFrameRate,
        0,
        100
      )
    );

  return {
    horizontalMovement:
      Math.round(
        horizontalMovement * 100
      ),

    verticalMovement:
      Math.round(
        verticalMovement * 100
      ),

    score,
    confidence
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
  const activeItems =
  trimTrailingStillFrames(items);

/*
 * Every observation in the app uses the same
 * estimated launch, contact and finish frames.
 */
const swingWindow =
  detectSwingWindow(activeItems);

const valid =
  activeItems.filter(
      frame => frame.world
    );

  const source =
    valid.length >= 5
      ? valid
      : activeItems;

  const kneeAngles = [];

  for (const frame of source) {
    const points =
      frame.world ||
      frame.landmarks;

    const leftKneeAngle = angle(
      points[23],
      points[25],
      points[27]
    );

    const rightKneeAngle = angle(
      points[24],
      points[26],
      points[28]
    );

    kneeAngles.push(
      Math.min(
        leftKneeAngle,
        rightKneeAngle
      )
    );
  }

  const kneeBend =
    Math.round(
      kneeAngles.reduce(
        (sum, value) =>
          sum + value,
        0
      ) / kneeAngles.length
    );

  const frontLeg =
  calculateFrontLegStability(
    activeItems,
    swingWindow
  );

  const headStability =
  calculateHeadStability(
    activeItems,
    swingWindow
  );

  const metricScores = {
    knee:
      scoreNear(
        kneeBend,
        105,
        45
      ),

    frontLeg:
      frontLeg.score,

    head:
      headStability.score
  };

  const overall =
    Math.round(
      (
        metricScores.knee +
        metricScores.frontLeg +
        metricScores.head
      ) / 3
    );

  const issues = [
    {
      key: "head",

      score:
        metricScores.head,

      title:
        "Keep the head centered",

      why:
        "Excessive head movement can change the hitter's view of the ball and reduce balance and consistent contact.",

      one:
        "Allow the body to rotate underneath a quiet, centered head.",

      drill:
        "Head-Stability Tee Drill — 3 rounds of 5 swings while holding the finish."
    },

    {
      key: "frontLeg",

      score:
        metricScores.frontLeg,

      title:
        "Front leg needs to firm up",

      why:
        "The lead knee did not appear to create a firm base as the hitter approached contact.",

      one:
        "Land under control, then allow the front leg to firm while the back hip comes through.",

      drill:
        "Front-Leg Brace Drill — 3 rounds of 5 controlled swings."
    },

    {
      key: "knee",

      score:
        metricScores.knee,

      title:
        "Maintain athletic posture",

      why:
        "Maintaining athletic knee flex improves balance, posture, and efficient movement throughout the swing.",

      one:
        "Stay athletic from load through contact without standing up or collapsing.",

      drill:
        "Hold-the-Finish Tee Drill — 3 rounds of 5 swings."
    }
  ].sort(
    (a, b) =>
      a.score - b.score
  );

    return {
    overall,

    metricScores,

    swingWindow,

    metrics: [
      [
        "Head Stability",

        `${headStability.horizontalMovement}%`,

        metricScores.head,

        `${headStability.horizontalMovement}% horizontal and ${headStability.verticalMovement}% vertical movement from heel plant through contact`
      ],

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
        }${frontLeg.firmingChange}° firming from heel plant toward contact`
      ]
    ],

    issue:
      issues[0]
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

  const timeline =
    a.swingWindow;

  const launchTime =
    frames[timeline.launchIndex]?.time ?? 0;

  const heelPlantTime =
    frames[timeline.heelPlantIndex]?.time ?? 0;

  const contactTime =
    frames[timeline.contactIndex]?.time ?? 0;

  const finishTime =
    frames[timeline.finishIndex]?.time ?? 0;

    const summaryEl =
    document.getElementById(
      "summary"
    );

  summaryEl.innerHTML = `
    The swing was analyzed using
    ${frames.length} accepted tracking frames.
    Your strongest opportunity for improvement is
    "${a.issue.title}".

    <br><br>

    <strong>Developer Timeline</strong>

    <br>

    <button
      type="button"
      class="secondary"
      data-jump-time="${launchTime}"
    >
      Launch — ${launchTime.toFixed(2)}s
    </button>

    <button
      type="button"
      class="secondary"
      data-jump-time="${heelPlantTime}"
    >
      Heel Plant — ${heelPlantTime.toFixed(2)}s
    </button>

    <button
      type="button"
      class="secondary"
      data-jump-time="${contactTime}"
    >
      Contact — ${contactTime.toFixed(2)}s
    </button>

    <button
      type="button"
      class="secondary"
      data-jump-time="${finishTime}"
    >
      Finish — ${finishTime.toFixed(2)}s
    </button>

    <br><br>

    Heel Plant Confidence:
    <strong>
      ${timeline.heelPlantConfidence}%
    </strong>
  `;

  /*
   * Developer buttons allow us to inspect exactly
   * which video frame the detector selected for
   * each phase of the swing.
   */
  summaryEl
    .querySelectorAll(
      "[data-jump-time]"
    )
    .forEach(button => {
      button.addEventListener(
        "click",
        () => {
          const jumpTime =
            Number(
              button.dataset.jumpTime
            );

          if (
            !Number.isFinite(
              jumpTime
            )
          ) {
            return;
          }

          video.pause();

          video.currentTime =
            Math.min(
              jumpTime,
              Math.max(
                0,
                video.duration - 0.001
              )
            );

          drawNearestFrame(
            jumpTime
          );

          statusEl.textContent =
            `Developer review: ${button.textContent.trim()}`;
        }
      );
    });

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
