const express = require("express");
const router = express.Router();
const { spawn } = require("child_process");

router.post("/", async (req, res) => {
  const session = req.body.session;

  // Save session to CSV
  const fs = require("fs");
  const path = "ml/temp_input.csv";
  const header = "X,Y,Pressure,Duration,Orientation,Size\n";
  const rows = session.map(
    (row) =>
      `${row.accelX},${row.accelY},${row.touchPressure},${row.duration},0,0`
  );

  fs.writeFileSync(path, header + rows.join("\n"));

  // Run predict_multi.py
  const py = spawn("python", ["ml/predict_multi.py"]);
  let output = "";

  py.stdout.on("data", (data) => (output += data.toString()));
  py.stderr.on("data", (err) => console.error("❌", err.toString()));

  py.on("close", (code) => {
    const [svm1Str, svm2Str, lstmStr] = output.trim().split(',');
    const svm1_score = parseFloat(svm1Str);
    const svm2_score = parseFloat(svm2Str);
    const lstm_score = parseFloat(lstmStr);

    let risk = "low";
    if (lstm_score < 0.4) risk = "high";
    else if (svm1_score < 0.4 && svm2_score < 0.4) risk = "medium";
    else if (svm1_score < 0.4) risk = "low-medium";

    return res.json({ svm1_score, svm2_score, lstm_score, risk });
  });
});

module.exports = router;
