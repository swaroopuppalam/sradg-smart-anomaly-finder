import json
from transformers import pipeline

# Optionally replace with OpenAI call if needed
USE_HF = True

ROOT_CAUSE_MODEL = pipeline("text-generation", model="gpt2", max_length=100)

def load_feature_importance(path="/shared/feature_importance.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

def suggest_root_cause(input_data):
    top_features = load_feature_importance()
    sorted_feats = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]

    summary = " | ".join([f"{k}: {input_data.get(k, 'N/A')}" for k, _ in sorted_feats])

    if USE_HF:
        prompt = f"Given the anomaly with features: {summary}, suggest a possible root cause:"
        try:
            result = ROOT_CAUSE_MODEL(prompt, do_sample=True, temperature=0.7)[0]['generated_text']
            return result.strip().split(prompt)[-1].strip()
        except Exception as e:
            return f"LLM error: {e}"
    else:
        return "Root cause LLM not enabled"
