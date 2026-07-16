import pandas as pd
import re
import string

csv_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\notebooks\finetune-wav2vec2\train_set\val_split.csv"
drugs_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\text\drugs.txt"

def normalize_for_eval(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open(drugs_path, 'r', encoding='utf-8') as f:
    drugs_raw = [line.strip() for line in f if len(line.strip()) > 2]
    
drug_set = set([normalize_for_eval(w) for w in drugs_raw])

df = pd.read_csv(csv_path)
col_name = 'text' if 'text' in df.columns else ('transcription' if 'transcription' in df.columns else 'path')

found_drugs = set()
for text in df[col_name].fillna(""):
    norm_text = normalize_for_eval(text)
    for d in drug_set:
        if d in norm_text:
            found_drugs.add(d)

print("=== CÁC BIỆT DƯỢC CÓ TRONG val_split.csv ===")
for d in sorted(found_drugs):
    # Find original casing from drugs_raw
    orig = next((orig for orig in drugs_raw if normalize_for_eval(orig) == d), d)
    print(f"- {orig}")
    
print(f"Tổng số: {len(found_drugs)} loại biệt dược.")
