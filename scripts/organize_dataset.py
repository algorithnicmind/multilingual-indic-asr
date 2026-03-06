import os
import shutil
import glob
from pathlib import Path

# Configuration
LANGUAGES = ['english', 'hindi', 'odia']
BASE_DIR = Path('data/raw')

def flatten_directory(language):
    print(f"\nProcessing {language}...")
    lang_dir = BASE_DIR / language
    clips_dir = lang_dir / 'clips'
    
    # Ensure clips directory exists
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Find all audio files (recursively)
    audio_files = []
    audio_files.extend(list(lang_dir.rglob('*.flac')))
    audio_files.extend(list(lang_dir.rglob('*.wav')))
    audio_files.extend(list(lang_dir.rglob('*.mp3')))
    
    # Filter out files already in clips dir to avoid errors
    audio_files = [f for f in audio_files if clips_dir not in f.parents]

    print(f"  Found {len(audio_files)} audio files.")
    
    # Move audio files
    for file_path in audio_files:
        try:
            shutil.move(str(file_path), str(clips_dir / file_path.name))
        except shutil.Error:
            print(f"  Skipping duplicate: {file_path.name}")

    # 2. Find all transcript files (recursively)
    # OpenSLR transcripts often end in .txt
    txt_files = list(lang_dir.rglob('*.txt'))
    # Filter out files already in clips/ (if any)
    txt_files = [f for f in txt_files if clips_dir not in f.parents and f.name != 'validated.tsv']
    
    print(f"  Found {len(txt_files)} transcript files.")
    
    # Consolidate transcripts into a single validated.tsv
    all_entries = []
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id = parts[0]
                    transcript = parts[1]
                    
                    # Construct filename (usually file_id.flac or .wav)
                    # We check which extension actually exists in clips/
                    audio_filename = f"{file_id}.flac"
                    if not (clips_dir / audio_filename).exists():
                         audio_filename = f"{file_id}.wav"
                    
                    if (clips_dir / audio_filename).exists():
                         # format: client_id, path, sentence, up_votes, down_votes, age, gender, accent
                         # We fill dummy data for missing fields
                         entry = f"{file_id}\tclips/{audio_filename}\t{transcript}\t0\t0\t\t\t"
                         all_entries.append(entry)

    # Append to validated.tsv if we found new entries
    if all_entries:
        tsv_path = lang_dir / 'validated.tsv'
        
        # Write header if file doesn't exist
        if not tsv_path.exists():
            with open(tsv_path, 'w', encoding='utf-8') as f:
                f.write("client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\n")
        
        # Append entries
        with open(tsv_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(all_entries) + '\n')
            
        print(f"  Added {len(all_entries)} entries to {tsv_path}")
    else:
        print("  No matching transcripts found to add to TSV.")

    print(f"  Cleanup complete for {language}.\n")

if __name__ == "__main__":
    print("Starting dataset organization...")
    for lang in LANGUAGES:
        if (BASE_DIR / lang).exists():
            flatten_directory(lang)
        else:
            print(f"Skipping {lang} (directory not found)")
    print("Done! You can now delete the empty subfolders in data/raw/<lang>/ if you wish.")
