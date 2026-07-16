import os
import csv
import wave
import argparse
import sys

# Reconfigure stdout to UTF-8 if supported to prevent encoding crashes on Windows terminals.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_wav_duration(file_path):
    """
    Get the duration of a WAV file using the built-in wave module.
    Returns the duration in seconds.
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
        return f"{hours} hours {minutes} minutes {secs:.2f} seconds ({seconds:.2f} seconds)"
    elif minutes > 0:
        return f"{minutes} minutes {secs:.2f} seconds ({seconds:.2f} seconds)"
    else:
        return f"{secs:.2f} seconds"

def main():
    parser = argparse.ArgumentParser(description="Calculate the total duration of a WAV audio dataset.")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="notebooks/finetune-wav2vec2/train_set/train.csv",
        help="Path to the CSV file containing the audio list (default: notebooks/finetune-wav2vec2/train_set/train.csv)"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="notebooks/finetune-wav2vec2/train_set/wavs",
        help="Directory containing the actual .wav files (default: notebooks/finetune-wav2vec2/train_set/wavs)"
    )
    parser.add_argument(
        "--path_col", 
        type=str, 
        default="path",
        help="Name of the CSV column containing file paths (default: 'path')"
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    audio_dir = args.audio_dir
    path_col = args.path_col
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at: {csv_path}")
        print("Please specify the correct path using the --csv_path argument")
        sys.exit(1)
        
    print(f"[*] Reading list from: {csv_path}")
    print(f"[*] Audio search directory: {audio_dir}")
    
    total_duration = 0.0
    processed_count = 0
    missing_count = 0
    error_count = 0
    durations = []
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if path_col not in reader.fieldnames:
                print(f"[ERROR] Column '{path_col}' does not exist in the CSV file.")
                print(f"Available columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
            rows = list(reader)
            total_rows = len(rows)
            print(f"[+] Found {total_rows} data rows in the CSV. Starting scan...")
            
            for idx, row in enumerate(rows, 1):
                rel_path = row[path_col]
                filename = os.path.basename(rel_path)
                
                # Check several candidate paths.
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
                    print(f"   Progress: {idx}/{total_rows} ({idx/total_rows*100:.1f}%) | Found: {processed_count} | Missing: {missing_count}")
    except Exception as e:
        print(f"[ERROR] Could not read the CSV file: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("DETAILED STATISTICS:")
    print("="*60)
    print(f" Total rows in CSV:           {total_rows}")
    print(f" Successfully processed:      {processed_count}")
    print(f" Files not found:             {missing_count}")
    print(f" Files with format errors:    {error_count}")
    
    if processed_count > 0:
        avg_dur = sum(durations) / processed_count
        min_dur = min(durations)
        max_dur = max(durations)
        print(f" Total calculated duration:   {format_time(total_duration)}")
        print(f" Average duration:            {avg_dur:.2f} seconds / file")
        print(f" Shortest duration:           {min_dur:.2f} seconds")
        print(f" Longest duration:            {max_dur:.2f} seconds")
        
        if missing_count > 0:
            estimated_missing_duration = missing_count * avg_dur
            estimated_total_duration = total_duration + estimated_missing_duration
            print(f"\n[WARNING] There are {missing_count} missing audio files on this local machine.")
            print(f" * Estimated duration of missing portion: {format_time(estimated_missing_duration)}")
            print(f" * Estimated TOTAL duration of FULL DATASET: {format_time(estimated_total_duration)}")
    else:
        print("[ERROR] No WAV files were found or readable for duration calculation.")
    print("="*60)

if __name__ == "__main__":
    main()
