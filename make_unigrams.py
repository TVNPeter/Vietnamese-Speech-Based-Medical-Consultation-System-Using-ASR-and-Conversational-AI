import re

corpus_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\text\corpus.txt"
out_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\text\unigrams.txt"

unigrams = set()
with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        words = line.lower().split()
        for w in words:
            # Chỉ lấy các ký tự chữ cái/số
            w_clean = re.sub(r'[^\w]', '', w)
            if w_clean:
                unigrams.add(w_clean)

with open(out_path, 'w', encoding='utf-8') as f:
    for w in sorted(unigrams):
        f.write(w + '\n')

print(f"Đã tạo {out_path} với {len(unigrams)} unigrams duy nhất.")
