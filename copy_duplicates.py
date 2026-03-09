import pandas as pd
import os
import shutil
import csv
import uuid
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def check_free_space(target_dir, min_free_gb=50):
    """Checks if the drive containing target_dir has at least min_free_gb of free space."""
    target_path = os.path.abspath(target_dir)
    # Ensure the directory exists so we can get its drive stats
    os.makedirs(target_path, exist_ok=True)
    total, used, free = shutil.disk_usage(target_path)
    free_gb = free / (1024**3)
    return free_gb >= min_free_gb, free_gb

def process_file_size(path):
    """Helper for multiprocessing pool to get file sizes safely."""
    try:
        if os.path.exists(path):
            return (path, os.path.getsize(path))
    except Exception:
        pass
    return (path, None)

def main(catalog_path, output_dir, mapping_csv):
    print(f"Loading catalog {catalog_path}...")
    try:
        df = pd.read_csv(catalog_path)
    except FileNotFoundError:
        print(f"Error: Catalog not found at {catalog_path}")
        return
        
    print(f"Found {len(df)} total files in catalog.")
    
    locations = df['location'].dropna().tolist()
    
    print("Finding potential exact duplicates by matching file bounds (size in bytes)...")
    size_groups = defaultdict(list)
    
    for path in tqdm(locations, desc="Checking File Sizes"):
        path, size = process_file_size(path)
        if size is not None:
             size_groups[size].append(path)
    
    potential_dupes = {size: paths for size, paths in size_groups.items() if len(paths) > 1}
    total_potential_files = sum(len(paths) for paths in potential_dupes.values())
    
    print(f"Found {len(potential_dupes)} size groups containing {total_potential_files} potential duplicate files.")
    
    if total_potential_files == 0:
        print("No identical file sizes found. No exact duplicates exist.")
        return
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    map_csv_path = Path(mapping_csv)
    
    # Load previously copied files to support resuming
    already_copied = set()
    open_mode = 'w'
    if map_csv_path.exists():
        try:
            with open(map_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    already_copied.add(row['original_path'])
            open_mode = 'a'
            print(f"Found existing mapping file. Resuming... automatically skipping {len(already_copied)} already copied files.")
        except Exception:
            pass
            
    print(f"Copying files sequentially to {out_dir}...")
    
    files_copied = 0
    with open(map_csv_path, open_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if open_mode == 'w':
            writer.writerow(['local_path', 'original_path'])
        
        # Flatten all potential dupes
        files_to_copy = []
        for paths in potential_dupes.values():
            files_to_copy.extend(paths)
            
        for orig_path in tqdm(files_to_copy, desc="Copying Duplicate Files"):
            if orig_path in already_copied:
                continue
                
            # Safety check before every copy
            is_safe, free_gb = check_free_space(out_dir, min_free_gb=50)
            if not is_safe:
                print(f"\n[WARNING] Local drive has less than 50GB free ({free_gb:.2f}GB remaining).")
                print(f"Stopping copy process safely. Copied {files_copied} files so far.")
                print(f"You can proceed to run hash_duplicates.py on the files copied so far.")
                break
                
            try:
                unique_name = f"{uuid.uuid4().hex}_{os.path.basename(orig_path)}"
                local_path = out_dir / unique_name
                
                shutil.copy2(orig_path, local_path)
                writer.writerow([str(local_path), str(orig_path)])
                files_copied += 1
                
            except Exception as e:
                # If a file is unreadable, skip it
                pass

    print(f"\nDone! Copied {files_copied} files to local drive.")
    print(f"Mapping saved to: {map_csv_path}")
    print("Next step: Run hash_duplicates.py")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    
    DEFAULT_CATALOG = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog_enriched.csv"
    if not DEFAULT_CATALOG.exists():
        DEFAULT_CATALOG = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog.csv"
        
    DEFAULT_OUTDIR = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "copies"
    DEFAULT_MAPPING = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "copy_mapping.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG), help="Path to catalog CSV")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Path to save copied images")
    parser.add_argument("--mapping", type=str, default=str(DEFAULT_MAPPING), help="Path to save mapping CSV")
    args = parser.parse_args()
    
    main(args.catalog, args.outdir, args.mapping)
