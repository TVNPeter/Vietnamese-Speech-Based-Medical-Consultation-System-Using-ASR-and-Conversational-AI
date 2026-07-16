import os
import csv
import wave
import argparse
import sys

# Reconfigure stdout to utf-8 if supported to prevent encoding crashes on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_wav_duration(file_path):
    """
    Get duration of a WAV file using the built-in wave module.
    Returns duration in seconds.
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception:
        return None

def format_time(seconds):
    """Format seconds into hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours} giờ {minutes} phút {secs:.2f} giây ({seconds:.2f} giây)"
    elif minutes > 0:
        return f"{minutes} phút {secs:.2f} giây ({seconds:.2f} giây)"
    else:
        return f"{secs:.2f} giây"

def main():
    parser = argparse.ArgumentParser(description="Tính toán tổng thời lượng của dataset âm thanh WAV.")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="notebooks/finetune-wav2vec2/train_set/train.csv",
        help="Đường dẫn tới file CSV chứa danh sách audio (mặc định: notebooks/finetune-wav2vec2/train_set/train.csv)"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="notebooks/finetune-wav2vec2/train_set/wavs",
        help="Thư mục chứa các file .wav thực tế (mặc định: notebooks/finetune-wav2vec2/train_set/wavs)"
    )
    parser.add_argument(
        "--path_col", 
        type=str, 
        default="path",
        help="Tên cột chứa đường dẫn file trong file CSV (mặc định: 'path')"
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    audio_dir = args.audio_dir
    path_col = args.path_col
    
    if not os.path.exists(csv_path):
        print(f"[LOI] Khong tim thay file CSV tai: {csv_path}")
        print("Vui long chi dinh duong dan dung bang tham so --csv_path")
        sys.exit(1)
        
    print(f"[*] Dang doc danh sach tu: {csv_path}")
    print(f"[*] Thu muc tim kiem am thanh: {audio_dir}")
    
    total_duration = 0.0
    processed_count = 0
    missing_count = 0
    error_count = 0
    durations = []
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if path_col not in reader.fieldnames:
                print(f"[LOI] Cot '{path_col}' khong ton tai trong file CSV.")
                print(f"Cac cot hien co: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
            rows = list(reader)
            total_rows = len(rows)
            print(f"[+] Tim thay {total_rows} dong du lieu trong CSV. Bat dau quet...")
            
            for idx, row in enumerate(rows, 1):
                rel_path = row[path_col]
                filename = os.path.basename(rel_path)
                
                # Check different candidate paths
                candidate_paths = [
                    os.path.join(audio_dir, filename),
                    os.path.join(os.path.dirname(csv_path), rel_path),
                    rel_path
                ]
                
                full_path = None
                for cp in candidate_paths:
                    if os.path.exists(cp):
                        full_path = cp
                        break
                        
                if full_path is None:
                    missing_count += 1
                    continue
                    
                duration = get_wav_duration(full_path)
                if duration is not None:
                    total_duration += duration
                    durations.append(duration)
                    processed_count += 1
                else:
                    error_count += 1
                    
                if idx % 500 == 0 or idx == total_rows:
                    print(f"   Tien do: {idx}/{total_rows} ({idx/total_rows*100:.1f}%) | Da tim thay: {processed_count} | Thieu: {missing_count}")
    except Exception as e:
        print(f"[LOI] Khong the doc file CSV: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("KET QUA THONG KE CHI TIET:")
    print("="*60)
    print(f" Tong so dong trong CSV:      {total_rows}")
    print(f" So file xu ly thanh cong:    {processed_count}")
    print(f" So file khong tim thay:      {missing_count}")
    print(f" So file loi dinh dang:       {error_count}")
    
    if processed_count > 0:
        avg_dur = sum(durations) / processed_count
        min_dur = min(durations)
        max_dur = max(durations)
        print(f" Tong thoi luong tinh duoc:   {format_time(total_duration)}")
        print(f" Thoi luong trung binh:       {avg_dur:.2f} giay / file")
        print(f" Thoi luong ngan nhat:        {min_dur:.2f} giay")
        print(f" Thoi luong dai nhat:         {max_dur:.2f} giay")
        
        if missing_count > 0:
            estimated_missing_duration = missing_count * avg_dur
            estimated_total_duration = total_duration + estimated_missing_duration
            print(f"\n[CANH BAO] Co {missing_count} file am thanh bi thieu tren may cuc bo nay.")
            print(f" * Uoc tinh thoi luong cua phan bi thieu: {format_time(estimated_missing_duration)}")
            print(f" * Du kien TONG thoi luong FULL DATASET:  {format_time(estimated_total_duration)}")
    else:
        print("[LOI] Khong tim thay hoac khong doc duoc file WAV nao de tinh thoi luong.")
    print("="*60)

if __name__ == "__main__":
    main()
