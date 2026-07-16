import re

corpus_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\text\corpus.txt"
out_path = r"E:\Github\Vietnamese Speech-Based Medical  Consultation System Using ASR and  Conversational AI\text\unigrams.txt"

unigrams = set()
# \w+ sẽ match liên tiếp các ký tự chữ/số. Nếu gặp dấu phẩy, chấm... nó sẽ tách ra làm 2 từ độc lập.
pattern = re.compile(r'\w+')

with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        words = pattern.findall(line.lower())
        unigrams.update(words)

with open(out_path, 'w', encoding='utf-8') as f:
    for w in sorted(unigrams):
        f.write(w + '\n')

print(f"Đã tạo lại {out_path} với {len(unigrams)} unigrams duy nhất chuẩn xác hơn.")
