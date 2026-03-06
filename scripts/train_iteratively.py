
import os
import shutil
import subprocess
import time
import random
import yaml
from pathlib import Path

# Paths
BASE_DIR = Path.cwd()
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_MINI = BASE_DIR / "data" / "raw_mini"
DATA_COMPLETED = BASE_DIR / "data" / "completed_raw"
CONFIG_PATH = BASE_DIR / "config.yaml"

LANGUAGES = ['english', 'hindi', 'odia']
BATCH_SIZE = 50  # Number of files to process per iteration

def setup_directories():
    """Ensure raw_mini and completed_raw directories exist."""
    for lang in LANGUAGES:
        (DATA_MINI / lang / "clips").mkdir(parents=True, exist_ok=True)
        (DATA_COMPLETED / lang / "clips").mkdir(parents=True, exist_ok=True)

def get_files_from_tsv(tsv_path):
    """Read entries from a TSV file."""
    if not tsv_path.exists():
        return []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Skip header if present
    if lines and lines[0].startswith('client_id'):
        return lines[1:]
    return lines

def move_batch():
    """Move a batch of files from raw to raw_mini."""
    files_moved = False
    
    for lang in LANGUAGES:
        print(f"Processing {lang}...")
        raw_tsv = DATA_RAW / lang / "validated.tsv"
        mini_tsv = DATA_MINI / lang / "validated.tsv"
        completed_tsv = DATA_COMPLETED / lang / "validated.tsv"
        
        # Check what's already completed
        completed_path = DATA_COMPLETED / lang / "clips"
        completed_files = {f.name for f in completed_path.glob("*")}
        
        # Read all available lines
        lines = get_files_from_tsv(raw_tsv)
        if not lines:
            print(f"  No data found in {lang} TSV.")
            continue
            
        # Find unprocessed lines
        unprocessed_lines = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            filename = Path(parts[1]).name
            if filename not in completed_files:
                unprocessed_lines.append(line)
        
        if not unprocessed_lines:
            # Check if we are done
            print(f"  No new data for {lang}. All files processed.")
            continue
            
        # Take batch
        batch_lines = unprocessed_lines[:BATCH_SIZE]
            
        print(f"  Copying {len(batch_lines)} files to mini batch...")
        
        # Prepare mini TSV
        mini_entries = []
        header = "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\n"
        
        for line in batch_lines:
            parts = line.strip().split('\t')
            relative_path = parts[1] # clips/filename.wav
            filename = Path(relative_path).name
            
            src_audio = DATA_RAW / lang / "clips" / filename
            dst_audio = DATA_MINI / lang / "clips" / filename
            
            if src_audio.exists():
                try:
                    shutil.copy2(src_audio, dst_audio) # COPY to mini, don't move
                    mini_entries.append(line)
                except Exception as e:
                    print(f"    Error copying {filename}: {e}")
            else:
                print(f"    Warning: Source file not found: {filename}")
        
        # Write mini TSV
        with open(mini_tsv, 'w', encoding='utf-8') as f:
            f.write(header + ''.join(mini_entries))
            
        # We DO NOT touch raw_tsv anymore
            
        files_moved = True
        
    return files_moved

def cleanup_mini_batch():
    """Move processed files from raw_mini to completed_raw."""
    for lang in LANGUAGES:
        mini_clips = DATA_MINI / lang / "clips"
        completed_clips = DATA_COMPLETED / lang / "clips"
        mini_tsv = DATA_MINI / lang / "validated.tsv"
        completed_tsv = DATA_COMPLETED / lang / "validated.tsv"
        
        # Move audio files
        for audio_file in mini_clips.glob("*"):
            shutil.move(str(audio_file), str(completed_clips / audio_file.name))
            
        # Append TSV entries
        if mini_tsv.exists():
            with open(mini_tsv, 'r', encoding='utf-8') as f:
                new_lines = f.readlines()
            
            # Remove header from new lines if appending
            if new_lines and new_lines[0].startswith('client_id'):
                new_lines = new_lines[1:]
                
            # Create completed TSV if not exists
            if not completed_tsv.exists():
                with open(completed_tsv, 'w', encoding='utf-8') as f:
                    f.write("client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\n")
            
            with open(completed_tsv, 'a', encoding='utf-8') as f:
                f.write(''.join(new_lines))
            
            # Clear mini TSV
            os.remove(mini_tsv)

def update_config_path(target_dir):
    """Update config.yaml to point to the correct data directory."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    config['paths']['data']['raw'] = str(target_dir).replace("\\", "/")
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    setup_directories()
    
    # Point config to MINI dataset
    update_config_path(DATA_MINI)
    
    iteration = 1
    while True:
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}")
        print(f"{'='*50}")
        
        # 1. Fill Mini Batch
        has_data = move_batch()
        if not has_data:
            print("All data processed! Training complete.")
            break
            
        # 2. Run Data Preparation (on mini batch)
        print("\nStep 1: Preparing Data...")
        subprocess.run(["python", "scripts/prepare_data.py"], check=True)
        
        # 3. Run Training (on mini batch)
        print("\nStep 2: Training...")
        # We assume train_all.py loads the checkpoint automatically if it exists
        subprocess.run(["python", "scripts/train_all.py"], check=True)
        
        # 4. Cleanup (Move to Completed)
        print("\nStep 3: Cleaning up...")
        cleanup_mini_batch()
        
        # 5. Clear Processed Cache (CRITICAL FOR SPEED)
        if (BASE_DIR / "data" / "processed").exists():
            shutil.rmtree(BASE_DIR / "data" / "processed")
        
        iteration += 1
        
        # Optional: Pause between iterations
        time.sleep(2)

if __name__ == "__main__":
    main()
