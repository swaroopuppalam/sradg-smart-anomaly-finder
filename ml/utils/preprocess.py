import pandas as pd

DEFAULT_THRESHOLDS = {
    "quantity_threshold": 1.0,
    "price_threshold": 0.01,
    "balance_threshold": 1.0
}

def preprocess_dataset(df, filename, config=None):
    # Merge provided config with fallback defaults
    config = config or {}
    rules = {**DEFAULT_THRESHOLDS, **config.get("anomaly_rules", {})}

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect anomaly label or derive
    if "anomaly" in df.columns:
        df["label"] = df["anomaly"].apply(
            lambda x: 1 if str(x).strip().lower() == "yes" else 0 if str(x).strip().lower() == "no" else 0
        )
        print("✅ Converted 'anomaly' column to 'label' with fallback for missing or non-standard values.")
    elif "balance_difference" in df.columns:
        df["label"] = df["balance_difference"].apply(
            lambda x: 1 if pd.notnull(x) and abs(x) > rules["balance_threshold"] else 0
        )
    elif {"quantitydifference", "pricedifference"}.issubset(df.columns):
        df["label"] = (
            (df["quantitydifference"].abs() > rules["quantity_threshold"]) |
            (df["pricedifference"].abs() > rules["price_threshold"])
        ).astype(int)
    else:
        df["label"] = 0  # fallback if no anomaly logic works

    # Feature engineering
    if "gl_balance" in df.columns and "ihub_balance" in df.columns:
        df["balance_difference"] = (
            pd.to_numeric(df["gl_balance"], errors="coerce") -
            pd.to_numeric(df["ihub_balance"], errors="coerce")
        )

    # Select relevant columns for training
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    final_cols = numeric_cols + ["label"]
    return df[final_cols]
