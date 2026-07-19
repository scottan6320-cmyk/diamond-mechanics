# Diamond Mechanics Pose Alpha

This is the first working pose-analysis prototype.

## What it does

- Records or selects an iPhone swing video
- Runs Google MediaPipe Pose Landmarker in the browser
- Draws a 33-point skeletal overlay
- Samples the video on-device
- Produces preliminary coaching estimates for:
  - knee bend
  - hip rotation
  - horizontal hip movement
  - vertical hip movement
  - hip-shoulder alignment
  - time to contact
- Identifies the lowest-scoring area
- Produces one coaching focus and one corrective drill

## Important alpha limitations

- This is not yet validated laboratory biomechanics.
- Camera angle, distance, lighting, clothing, and body visibility affect results.
- Movement is normalized to body size rather than reported in inches.
- Hip rotation from a single 2D phone camera is an estimate.
- Scores are transparent prototype rules and will be revised using real player testing.
- The MediaPipe model downloads from the internet the first time the page opens.

## Recording setup

Use an open-side view, steady phone, good lighting, full body in frame, and one swing per clip.

## Install

Replace the root files in the GitHub Pages repository with:

- index.html
- styles.css
- app.js

Commit and refresh the published site.
