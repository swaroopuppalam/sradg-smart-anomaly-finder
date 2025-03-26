import json
from transformers import pipeline

EXPLANATION_MAP_PATH = "/shared/explanation_map.json"

REASON_LABELS = [
    "Price Mismatch",
    "Quantity Mismatch",
    "Timing Issue",
    "GL vs IHub Difference",
    "Data Entry Error",
    "System Sync Delay"
]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def clean_comment(raw_comment):
    try:
        if not raw_comment.strip().startswith("["):
            return raw_comment.strip()

        loaded = json.loads(raw_comment.replace("'", "\""))
        if isinstance(loaded, list):
            return " ".join(
                x["text"] if isinstance(x, dict) and "text" in x else str(x)
                for x in loaded
            )
    except Exception as e:
        print(f"⚠️ Failed to parse COMMENT: {raw_comment} → {e}")
    return str(raw_comment).strip()

def extract_reason_llm(comment: str) -> str:
    if not comment or not isinstance(comment, str):
        return "Unknown Reason"
    result = classifier(comment, REASON_LABELS)
    if result["scores"][0] < 0.7:
        return "New Reason"
    return result["labels"][0]

def generate_reasoning_map_llm(data, save_path=EXPLANATION_MAP_PATH):
    mapping = {}
    print("\n🔎 Generating explanation_map.json with LLM reasoning...")

    for idx, row in enumerate(data):
        try:
            raw_comment = row.get("COMMENT") or row.get("Comments") or row.get("comment")
            if not raw_comment or not isinstance(raw_comment, str):
                print(f"⚠️ Row {idx}: No valid comment → Skipped")
                continue

            comment = clean_comment(raw_comment)
            if not comment or not isinstance(comment, str):
                print(f"⚠️ Row {idx}: Cleaned comment invalid → Skipped")
                continue

            print(f"\n➡️ Raw COMMENT [{idx}]: {raw_comment}")
            print(f"🧽 Cleaned COMMENT: {comment}")

            reason = extract_reason_llm(comment)
            print(f"🧠 LLM classified: '{comment}' → {reason}")

            numeric_values = []
            for k, v in row.items():
                try:
                    num = round(float(v), 3)
                    numeric_values.append(num)
                except:
                    continue

            key = str(tuple(numeric_values))  # Removed __comment from key to improve match coverage
            if key:
                mapping[key] = reason
                print(f"✅ Added to map → Key: {key[:60]}..., Reason: {reason}")

        except Exception as e:
            print(f"⚠️ Skipped row {idx} due to error: {e}")

    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n📌 Saved explanation_map.json with {len(mapping)} entries at {save_path}")
