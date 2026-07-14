# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import json
import re

def clean_parentheses(text):
    # Xử lý dấu ngoặc đơn (...)
    matches = list(re.finditer(r'\((.*?)\)', text))
    for m in reversed(matches):
        content = m.group(1).strip()
        start, end = m.span()
        
        # 1. Nếu bên trong ngoặc chỉ là 1 acronym đơn giản (ví dụ: (DHF), (THF)) -> Xóa bỏ hoàn toàn
        if re.match(r'^[A-Z0-9\s-]{1,8}$', content):
            text = text[:start] + text[end:]
            continue
            
        # 2. Nếu bên trong ngoặc là cụm ghép giữa từ thường và viết tắt phân tách bởi gạch ngang, xuyệt hoặc phẩy
        # ví dụ: (tetrahydrofolate - THF), (THF / tetrahydrofolate), (dihydrofolate, DHF)
        parts = re.split(r'[-/,]', content)
        parts = [p.strip() for p in parts]
        
        has_acronym = False
        new_content_parts = []
        for p in parts:
            if re.match(r'^[A-Z0-9\s]{1,8}$', p):
                has_acronym = True
                continue  # Bỏ qua phần viết tắt
            else:
                new_content_parts.append(p)
                
        if has_acronym and new_content_parts:
            remaining_content = " ".join(new_content_parts).strip()
            if remaining_content:
                text = text[:start] + " " + remaining_content + " " + text[end:]
            else:
                text = text[:start] + text[end:]
        else:
            # 3. Nếu là giải nghĩa bình thường không chứa viết tắt tách riêng -> Giữ lại nội dung và bỏ dấu ngoặc
            text = text[:start] + " " + content + " " + text[end:]
            
    # Xử lý tương tự với dấu ngoặc vuông [...]
    matches = list(re.finditer(r'\[(.*?)\]', text))
    for m in reversed(matches):
        content = m.group(1).strip()
        start, end = m.span()
        if re.match(r'^[A-Z0-9\s-]{1,8}$', content):
            text = text[:start] + text[end:]
            continue
            
        parts = re.split(r'[-/,]', content)
        parts = [p.strip() for p in parts]
        
        has_acronym = False
        new_content_parts = []
        for p in parts:
            if re.match(r'^[A-Z0-9\s]{1,8}$', p):
                has_acronym = True
                continue
            else:
                new_content_parts.append(p)
                
        if has_acronym and new_content_parts:
            remaining_content = " ".join(new_content_parts).strip()
            if remaining_content:
                text = text[:start] + " " + remaining_content + " " + text[end:]
            else:
                text = text[:start] + text[end:]
        else:
            text = text[:start] + " " + content + " " + text[end:]
            
    return text

def split_into_sentences(text):
    # Trước khi tách câu, làm sạch ngoặc
    text = clean_parentheses(text)
    
    # Tách câu dựa trên dấu chấm, chấm hỏi, chấm than
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    final_sentences = []
    for s in sentences:
        final_sentences.extend(s.split('\n'))
    return final_sentences

def normalize_sentence(sentence):
    # 1. Chuyển thành chữ thường
    sentence = sentence.lower()
    
    # 2. Xóa các ký tự không phải là chữ cái, số hoặc khoảng trắng
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    
    # 3. Xóa dấu gạch dưới
    sentence = sentence.replace('_', ' ')
    
    # 4. Gộp nhiều khoảng trắng thành một khoảng trắng duy nhất
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence.strip()

def main():
    input_file = 'randomqa.txt'
    output_file = 'corpus.txt'
    
    print(f"Start reading from {input_file}...")
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            q = data.get('question', '')
            a = data.get('answer', '')
            
            for text in [q, a]:
                if not text:
                    continue
                sentences = split_into_sentences(text)
                for sent in sentences:
                    norm = normalize_sentence(sent)
                    if norm and len(norm.split()) > 1:
                        fout.write(norm + '\n')
                        count += 1
                        
    print(f"Done! Wrote {count} sentences to {output_file}.")

if __name__ == '__main__':
    main()
