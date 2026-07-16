import pandas as pd
import glob
import re
import os
from num2words import num2words

def replace_numbers(match):
    num_str = match.group(0).replace(',', '.')
    try:
        if '.' in num_str:
            word = num2words(float(num_str), lang='vi')
        else:
            word = num2words(int(num_str), lang='vi')
        if num_str.endswith('5') and 'phẩy' in word:
            word = word.replace(' mươi', '')
        return ' ' + word + ' '
    except Exception as e:
        return match.group(0)

def normalize_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Sửa lỗi dịch thuật/ký tự đặc biệt trước
    text = text.replace('dotyczące', ' liên quan đến ')
    text = text.replace('ฤทธ', ' tác dụng ')
    
    # Ký tự Hy Lạp / ký hiệu khoa học
    text = text.replace('½', ' một phần hai ')
    text = text.replace('β', ' beta ')
    text = text.replace('α', ' alpha ')
    text = text.replace('κ', ' kappa ')
    text = text.replace('µ', ' micro ')
    text = text.replace('μ', ' micro ')
    
    # Chỉ số trên / dưới
    text = text.replace('²', ' vuông ')
    text = text.replace('³', ' khối ')
    text = text.replace('₂', ' hai ')
    text = text.replace('₅', ' năm ')

    # Ký hiệu phần trăm
    text = text.replace('%', ' phần trăm ')
    
    # Dấu gạch ngang giữa 2 số có thể đọc là "đến"
    text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 đến \2', text)
    
    # Chuyển đổi số thành chữ
    text = re.sub(r'\d+(?:[.,]\d+)?', replace_numbers, text)
    
    # Loại bỏ các ký tự đặc biệt, chỉ giữ lại chữ cái tiếng Việt, tiếng Anh và khoảng trắng
    text = re.sub(r'[^a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]', ' ', text)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    csv_files = glob.glob('train_set/*.csv')
    print(f"Found CSV files: {csv_files}")
    for file in csv_files:
        print(f"Processing {file}...")
        try:
            df = pd.read_csv(file)
            if 'text' in df.columns:
                df['text'] = df['text'].apply(normalize_text)
                df.to_csv(file, index=False, encoding='utf-8')
                print(f"  -> Saved {file}")
            else:
                print(f"  -> No 'text' column found in {file}")
        except Exception as e:
            print(f"  -> Error processing {file}: {e}")

if __name__ == "__main__":
    main()
