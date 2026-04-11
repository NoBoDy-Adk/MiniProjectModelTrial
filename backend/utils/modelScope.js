const path = require("path");

const ROOT_DIR = path.join(__dirname, "..");
const ML_DIR = path.join(ROOT_DIR, "ml");
const PROFILE_ROOT_DIR = path.join(ML_DIR, "user_profiles");

function sanitizeAccountNo(accountNo) {
  const raw = String(accountNo || "").trim();
  if (!raw) {
    return "shared";
  }
  return raw.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 128) || "shared";
}

function resolveModelScope(accountNo) {
  const scopeId = sanitizeAccountNo(accountNo);
  const scopeDir = path.join(PROFILE_ROOT_DIR, scopeId);

  return {
    scopeId,
    scopeDir,
    tempInputPath: path.join(scopeDir, "temp_input.csv"),
    historyPath: path.join(scopeDir, "history_input.csv"),
    referencePath: path.join(scopeDir, "reference_session.csv"),
    svmSeqPath: path.join(scopeDir, "svm_tier_1_sequence.pkl"),
    svmStatPath: path.join(scopeDir, "svm_tier_2_statistical.pkl"),
    lstmPath: path.join(scopeDir, "lstm_classifier.pt"),
  };
}

module.exports = {
  resolveModelScope,
};
