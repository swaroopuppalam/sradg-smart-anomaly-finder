import pandas as pd
import joblib
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.utils import resample
from joblib import load
import traceback
from utils.preprocess import preprocess_dataset  # ✅ NEW import

print("🔄 Retraining Model from Feedback (Integrated Data Mode)...")

# --------------------------
# 🔹 Paths
# --------------------------
FEEDBACK_PATH = "/ui/feedback.json"
MODEL_PATH = "/app/model.pkl"
TRAINING_LOGS = "/shared/training_logs.json"
CONFIG_PATH = "/ml/config.json"
FEATURE_IMPORTANCE_PATH = "/shared/feature_importance.json"
YTEST_OUTPUT_PATH = "/shared/y_true_pred.json"
EXPLANATION_MAP_PATH = "/shared/explanation_map.json"
EXPLANATION_FEATURES_PATH = "/shared/explanation_features.json"
DATA_DIR = "/ml"

# --------------------------
# 🔹 Load model (if exists)
# --------------------------
def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Failed to load model from {path}: {e}")
        traceback.print_exc()
        return None

model = safe_load_model(MODEL_PATH)

# --------------------------
# 🔹 Load anomaly config
# --------------------------
DEFAULT_CONFIG = {"quantity_threshold": 1.0, "price_threshold": 0.01, "balance_threshold": 1.0}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        print(f"🧪 Loaded anomaly thresholds: {config}")
    except:
        config = {"anomaly_rules": DEFAULT_CONFIG}
        print("⚠️ Failed to load config. Using defaults.")
else:
    config = {"anomaly_rules": DEFAULT_CONFIG}
    print("⚠️ Config file not found. Using defaults.")

# --------------------------
# 🔹 Load data safely
# --------------------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"⚠️ Warning: Could not read {path}: {e}")
        return pd.DataFrame()

# --------------------------
# 🔹 Process all CSVs dynamically from /ml
# --------------------------
dataframes = []
raw_anomalies = []  # ✅ NEW for explanation map
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".csv"):
        path = os.path.join(DATA_DIR, fname)
        df_raw = safe_read_csv(path)
        if not df_raw.empty:
            processed = preprocess_dataset(df_raw, fname, config)
            if not processed.empty:
                dataframes.append(processed)

            if "Anomaly" in df_raw.columns and "label" not in df_raw.columns:
                df_raw["label"] = df_raw["Anomaly"].apply(lambda x: 1 if str(x).strip().lower() in ["1", "yes", "true"] else 0)
                print(f"✅ Converted 'anomaly' column to 'label' with fallback for missing or non-standard values.")

            comment_col = None
            for col in df_raw.columns:
                if col.lower() in ["comment", "comments"]:
                    comment_col = col
                    break

            if "label" in df_raw.columns and comment_col:
                anomaly_rows = df_raw[df_raw["label"] == 1]
                print(f"🔎 Found {len(anomaly_rows)} anomaly rows with comments in {fname}")
                for i, row in anomaly_rows.iterrows():
                    raw_comment = row[comment_col]
                    print(f"➡️ Raw COMMENT [{i}]: {raw_comment}")
                    if pd.isna(raw_comment):
                        continue
                    row_dict = row.to_dict()
                    row_dict["COMMENT"] = raw_comment  # ✅ Unified key for LLM
                    raw_anomalies.append(row_dict)

# --------------------------
# 🔹 Load & process feedback.json
# --------------------------
if os.path.exists(FEEDBACK_PATH):
    try:
        feedback = json.load(open(FEEDBACK_PATH))
        df_fb = pd.DataFrame(feedback)
        fb_data = pd.json_normalize(df_fb["input"])
        fb_data["label"] = df_fb["feedback"].apply(lambda x: 1 if x == "Yes" else 0)
        fb_data = fb_data.select_dtypes(include=["number"])
        fb_data["label"] = fb_data["label"].astype(int)
        if not fb_data.empty:
            dataframes.append(fb_data)
    except Exception as e:
        print(f"⚠️ Feedback parse error: {e}")

# --------------------------
# 🔹 Merge data for training
# --------------------------
if not dataframes:
    print("❌ No valid data found. Cannot train model.")
    exit()

full_data = pd.concat(dataframes, ignore_index=True)
X = full_data.drop("label", axis=1).select_dtypes(include=["number"])
y = full_data["label"]

# --------------------------
# 🔹 Balance data
# --------------------------
normal_df = full_data[full_data["label"] == 0]
anomaly_df = full_data[full_data["label"] == 1]
if len(anomaly_df) < len(normal_df):
    anomaly_df = resample(anomaly_df, replace=True, n_samples=len(normal_df), random_state=42)

train_data = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=42).reset_index(drop=True)
X_resampled = train_data.drop("label", axis=1).select_dtypes(include=["number"])
y_resampled = train_data["label"]

# --------------------------
# 🔹 Train/test split & training
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

if model:
    print("🔁 Incrementally Training the Existing Model...")
    model.fit(X_train, y_train)
else:
    print("📌 Training a New Model from Scratch...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 3},
        random_state=42
    )
    model.fit(X_train, y_train)

# --------------------------
# 🔹 Evaluation
# --------------------------
print("📊 Model Performance After Retraining:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open(YTEST_OUTPUT_PATH, "w") as f:
    json.dump({"y_test": y_test.tolist(), "y_pred": y_pred.tolist()}, f, indent=2)
print("📉 Saved y_test and y_pred to /shared/y_true_pred.json")

# --------------------------
# 🔹 Save model
# --------------------------
joblib.dump(model, MODEL_PATH)
print("✅ Model retrained and saved to", MODEL_PATH)

# --------------------------
# 🔹 Save training logs
# --------------------------
entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": round(accuracy_score(y_test, y_pred), 3),
    "precision_0": round(precision_score(y_test, y_pred, pos_label=0), 3),
    "recall_0": round(recall_score(y_test, y_pred, pos_label=0), 3),
    "precision_1": round(precision_score(y_test, y_pred, pos_label=1), 3),
    "recall_1": round(recall_score(y_test, y_pred, pos_label=1), 3),
    "y_test": y_test.tolist(),
    "y_pred": y_pred.tolist()
}
logs = json.load(open(TRAINING_LOGS, "r")) if os.path.exists(TRAINING_LOGS) else []
logs.append(entry)
with open(TRAINING_LOGS, "w") as f:
    json.dump(logs, f, indent=2)


# --------------------------
# 🔹 Save feature importance
# --------------------------
feature_importance = dict(zip(X_resampled.columns, model.feature_importances_))
with open(FEATURE_IMPORTANCE_PATH, "w") as f:
    json.dump(feature_importance, f, indent=2)
print(f"📊 Saved feature importances to {FEATURE_IMPORTANCE_PATH}")

# --------------------------
# 🔹 Save feature list
# --------------------------
with open("/shared/feature_list.json", "w") as f:
    json.dump(list(X_resampled.columns), f)
print("🧾 Saved feature list to /shared/feature_list.json")

# --------------------------
# 🔹 Save explanation features
# --------------------------
with open(EXPLANATION_FEATURES_PATH, "w") as f:
    json.dump(list(X_resampled.columns), f)
print("🧾 Saved explanation feature list to /shared/explanation_features.json")

# --------------------------
# 🔹 Generate explanation_map from COMMENT field
# --------------------------
try:
    from utils.reasoning_llm import generate_reasoning_map_llm  # ✅ Import the new logic

    try:
        print("\n🔎 Generating explanation_map.json with LLM reasoning...")
        generate_reasoning_map_llm(raw_anomalies)
    except Exception as e:
        print("⚠️ Failed to generate explanation_map using LLM:", e)

    print("📌 Saved explanation_map.json from COMMENT field.")
except Exception as e:
    print("⚠️ Failed to generate explanation map:", e)
