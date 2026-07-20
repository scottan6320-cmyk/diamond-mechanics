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

  const stopAt = Math.min(video.duration, 8);
  const originalMuted = video.muted;
  const originalRate = video.playbackRate;
  const originalControls = video.controls;

  statusEl.textContent = "Analyzing while the video plays…";

  try {
    video.pause();
    video.currentTime = 0;
    video.muted = true;
    video.playbackRate = 0.5;
    video.controls = false;

    await waitForVideoReady();

    let lastProcessedTime = -1;
    let finished = false;

    await new Promise(async (resolve, reject) => {
      const timeoutId = setTimeout(() => {
        if (!finished) reject(new Error("Video playback timed out."));
      }, Math.max(15000, stopAt * 5000));

      const finish = () => {
        if (finished) return;
        finished = true;
        clearTimeout(timeoutId);
        video.pause();
        resolve();
      };

      const processFrame = () => {
        if (finished) return;

        const current = video.currentTime;
        if (current >= stopAt || video.ended) {
          finish();
          return;
        }

        // Analyze about 12 frames per second to keep iPhone Safari responsive.
        if (lastProcessedTime < 0 || current - lastProcessedTime >= 1 / 12) {
          lastProcessedTime = current;
          const tracked = detectLockedHitter(Math.round(current * 1000));

          if (tracked && isReliablePose(tracked.landmarks)) {
            const hip = midpoint(tracked.landmarks[23], tracked.landmarks[24]);
            const jump = lastAcceptedHip ? distance2D(hip, lastAcceptedHip) : 0;

            if (!lastAcceptedHip || jump < 0.16) {
              frames.push({
                time: current,
                landmarks: tracked.landmarks,
                world: tracked.world
              });
              lastAcceptedHip = hip;
            }
          }

          progress.value = Math.min(100, (current / stopAt) * 100);
        }

        if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
          video.requestVideoFrameCallback(processFrame);
        } else {
          requestAnimationFrame(processFrame);
        }
      };

      video.addEventListener("ended", finish, { once: true });

      try {
        await video.play();
        processFrame();
      } catch (error) {
        clearTimeout(timeoutId);
        reject(new Error("Safari blocked video playback. Tap the video play button once, pause it, then tap Analyze Swing."));
      }
    });

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

function distance2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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

function calculateMetrics(items) {
  const valid = items.filter(f => f.world);
  const source = valid.length >= 5 ? valid : items;

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
  const horizontalHip = percentileRange(hipCenters.map(p=>p.x))/bodyScale;
  const verticalHip = percentileRange(hipCenters.map(p=>p.y))/bodyScale;
  const alignment = clamp(100 - Math.abs(hipRotation-shoulderRotation)*1.1,0,100);
  const timeToContact = estimateTimeToContact(source);

  const metricScores = {
    knee: scoreNear(kneeBend,105,45),
    rotation: scoreBelow(Math.abs(105-hipRotation),15,70),
    horizontal: scoreBelow(horizontalHip,0.10,0.55),
    vertical: scoreBelow(verticalHip,0.08,0.40),
    alignment: Math.round(alignment),
    timing: scoreBelow(timeToContact,0.22,0.65)
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
    {key:"rotation", score:metricScores.rotation, title:"Limited hip rotation",
      why:"Insufficient lower-half rotation may limit energy transfer into the bat.",
      one:"Finish with the belt buckle turned toward the pitcher.",
      drill:"Walk-Through Rotation Drill — 3 rounds of 5 swings."},
    {key:"alignment", score:metricScores.alignment, title:"Hip-shoulder sequence needs work",
      why:"When hips and shoulders rotate inefficiently, energy transfer and timing can suffer.",
      one:"Let the hips begin while the shoulders stay closed a moment longer.",
      drill:"Separation Drill — 3 rounds of 5 deliberate swings."},
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
      ["Knee Bend",`${kneeBend}°`,metricScores.knee,"Estimated average"],
      ["Hip Rotation",`${hipRotation}°`,metricScores.rotation,"Estimated range"],
      ["Horizontal Hip Movement",`${(horizontalHip*100).toFixed(0)}%`,metricScores.horizontal,"Relative to torso"],
      ["Vertical Hip Movement",`${(verticalHip*100).toFixed(0)}%`,metricScores.vertical,"Relative to torso"],
      ["Hip-Shoulder Alignment",`${Math.round(alignment)}%`,metricScores.alignment,"Rotation similarity"],
      ["Time to Contact",`${timeToContact.toFixed(2)}s`,metricScores.timing,"Estimated motion window"]
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
    a.overall >= 75 ? "Strong movement profile" :
    a.overall >= 55 ? "Useful starting point" : "Clear opportunity to improve";
  document.getElementById("summary").textContent =
    "Only frames that passed the tracking-quality check were included.";

  metricsEl.innerHTML = a.metrics.map(([name,value,score,note]) => `
    <div class="metric ${score>=75?"good":score>=55?"warn":"bad"}">
      <h3>${name}</h3>
      <strong>${value}</strong>
      <small>Score ${score}/100 • ${note}</small>
    </div>`).join("");

  document.getElementById("primaryIssue").textContent = a.issue.title;
  document.getElementById("why").textContent = a.issue.why;
  document.getElementById("oneThing").textContent = a.issue.one;
  document.getElementById("drill").textContent = a.issue.drill;
  report.scrollIntoView({behavior:"smooth"});
}

init();
