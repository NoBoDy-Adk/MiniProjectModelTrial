const express = require("express");
const path = require("path");
const { spawn } = require("child_process");

const router = express.Router();

function clamp01(value, fallback = 0.5) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }

  if (numeric < 0) {
    return 0;
  }

  if (numeric > 1) {
    return 1;
  }

  return numeric;
}

function normalizeByRange(value, min, max, fallback = 0.5) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }

  const range = max - min;
  if (!Number.isFinite(range) || range <= 0) {
    return fallback;
  }

  return clamp01((numeric - min) / range, fallback);
}

function normalizeSession(session) {
  const safeSession = Array.isArray(session) ? session : [];
  const pressureValues = safeSession
    .map((row) => Number(row.touchPressure))
    .filter((value) => Number.isFinite(value) && value >= 0);

  const minPressure = pressureValues.length ? Math.min(...pressureValues) : 0;
  const maxPressure = pressureValues.length ? Math.max(...pressureValues) : 1;
  const parsed = safeSession.map((row) => ({
    x: Number(row.accelX),
    y: Number(row.accelY),
    duration: Number(row.duration),
    pressure: Number(row.touchPressure),
  }));

  const steps = parsed.map((current, index) => {
    const previous = index > 0 ? parsed[index - 1] : null;
    const x = Number.isFinite(current.x) ? current.x : 0;
    const y = Number.isFinite(current.y) ? current.y : 0;
    const previousX = previous && Number.isFinite(previous.x) ? previous.x : x;
    const previousY = previous && Number.isFinite(previous.y) ? previous.y : y;
    const dx = x - previousX;
    const dy = y - previousY;
    const deltaDistance = Math.sqrt(dx * dx + dy * dy);

    const rawDuration = Number.isFinite(current.duration) ? current.duration : 0;
    const previousDuration =
      previous && Number.isFinite(previous.duration) ? previous.duration : 0;
    const deltaDuration = rawDuration > previousDuration ? rawDuration - previousDuration : rawDuration;
    const safeDeltaDuration = Math.max(deltaDuration, 1);
    const velocity = deltaDistance / safeDeltaDuration;

    return {
      dx,
      dy,
      deltaDistance,
      deltaDuration: safeDeltaDuration,
      velocity,
      pressure: current.pressure,
    };
  });

  const absDxMax = Math.max(
    ...steps.map((step) => Math.abs(step.dx)).filter((value) => Number.isFinite(value)),
    1
  );
  const absDyMax = Math.max(
    ...steps.map((step) => Math.abs(step.dy)).filter((value) => Number.isFinite(value)),
    1
  );
  const maxDeltaDuration = Math.max(
    ...steps.map((step) => step.deltaDuration).filter((value) => Number.isFinite(value)),
    1
  );
  const maxVelocity = Math.max(
    ...steps.map((step) => step.velocity).filter((value) => Number.isFinite(value)),
    1e-6
  );
  const totalDistance = steps.reduce(
    (accumulator, step) => accumulator + (Number.isFinite(step.deltaDistance) ? step.deltaDistance : 0),
    0
  );
  let distancePrefix = 0;

  return steps.map((step, index) => {
    const pressure =
      maxPressure > 1
        ? normalizeByRange(step.pressure, minPressure, maxPressure)
        : clamp01(step.pressure, 0.5);
    const x = clamp01((step.dx / absDxMax + 1) / 2, 0.5);
    const y = clamp01((step.dy / absDyMax + 1) / 2, 0.5);
    const duration = clamp01(step.deltaDuration / maxDeltaDuration, 0);
    const orientation = clamp01(step.velocity / maxVelocity, 0.5);
    distancePrefix += Number.isFinite(step.deltaDistance) ? step.deltaDistance : 0;
    const size =
      totalDistance > 0
        ? clamp01(distancePrefix / totalDistance, 0.5)
        : clamp01((index + 1) / Math.max(steps.length, 1), 0.5);

    return {
      X: x,
      Y: y,
      Pressure: pressure,
      Duration: duration,
      Orientation: orientation,
      Size: size,
    };
  });
}

router.post("/", async (req, res) => {
  const normalizedSession = normalizeSession(req.body.session);

  if (!normalizedSession.length) {
    return res.status(400).json({ message: "Session data is required." });
  }

  const fs = require("fs");
  const inputPath = path.join(__dirname, "../ml/temp_input.csv");
  const header = "X,Y,Pressure,Duration,Orientation,Size\n";
  const rows = normalizedSession.map(
    (row) =>
      `${row.X},${row.Y},${row.Pressure},${row.Duration},${row.Orientation},${row.Size}`
  );

  fs.writeFileSync(inputPath, header + rows.join("\n"));

  const py = spawn("python", ["ml/predict_multi.py"], {
    cwd: path.join(__dirname, ".."),
  });

  let output = "";
  let errorOutput = "";

  py.stdout.on("data", (data) => {
    output += data.toString();
  });

  py.stderr.on("data", (data) => {
    const message = data.toString();
    errorOutput += message;
    console.error("predict error:", message);
  });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        message: "Model prediction failed.",
        detail: errorOutput.trim() || `Predict script exited with code ${code}`,
      });
    }

    const [svm1Str, svm2Str, lstmStr] = output.trim().split(",");
    const svm1_score = parseFloat(svm1Str);
    const svm2_score = parseFloat(svm2Str);
    const lstm_score = parseFloat(lstmStr);

    if (
      !Number.isFinite(svm1_score) ||
      !Number.isFinite(svm2_score) ||
      !Number.isFinite(lstm_score)
    ) {
      return res.status(500).json({
        message: "Model prediction returned invalid scores.",
        detail: output.trim() || errorOutput.trim() || "No score output received.",
      });
    }

    let risk = "low";
    if (lstm_score < 0.4) {
      risk = "high";
    } else if (svm1_score < 0.4 && svm2_score < 0.4) {
      risk = "medium";
    } else if (svm1_score < 0.4) {
      risk = "low-medium";
    }

    return res.json({ svm1_score, svm2_score, lstm_score, risk });
  });
});

module.exports = router;
