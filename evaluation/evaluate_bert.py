import pandas as pd
from bert_score import score

# ===== CONFIG =====
INPUT_CSV = ""
OUTPUT_CSV = ""
MODEL_TYPE = "models/BERT"   # 
LANG = "vi"  
# ===== LOAD DATA =====
df = pd.read_csv(INPUT_CSV)

df = df.fillna("")

# ===== PREPARE TEXT =====
preds = df["predicted"].astype(str).tolist()
refs = df["output"].astype(str).tolist()
docs = df["input"].astype(str).tolist()

# ===== BERTScore-ref =====
print("Calculating BERTScore-ref...")
P_ref, R_ref, F1_ref = score(
    preds,
    refs,
    model_type=MODEL_TYPE,
    num_layers=9,
    lang=LANG,
    verbose=True
)

# ===== BERTScore-doc =====
print("Calculating BERTScore-doc...")
P_doc, R_doc, F1_doc = score(
    preds,
    docs,
    model_type=MODEL_TYPE,
    lang=LANG,
    num_layers=9,
    verbose=True
)

# ===== SAVE RESULTS =====
metrics = {
    "metric": ["BERTScore-ref", "BERTScore-doc"],
    "precision": [P_ref.mean().item(), P_doc.mean().item()],
    "recall": [R_ref.mean().item(), R_doc.mean().item()],
    "f1": [F1_ref.mean().item(), F1_doc.mean().item()]
}

summary_df = pd.DataFrame(metrics)

# ===== SAVE & PRINT =====
summary_df.to_csv(OUTPUT_CSV, index=False)