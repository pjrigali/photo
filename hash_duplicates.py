import pandas as pd
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import os
import csv
from multiprocessing import Pool, cpu_count

def process_file_hash(path):
    """Helper for multiprocessing pool to get file MD5 hashes safely."""
    md5 = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                md5.update(chunk)
        return (path, md5.hexdigest())
    except Exception:
        return (path, None)

def main(mapping_csv, output_path, workers=None, cleanup=True):
    print(f"Loading local mapping {mapping_csv}...")
    try:
        df = pd.read_csv(mapping_csv)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_csv}")
        print("Please run copy_duplicates.py first!")
        return
        
    print(f"Found {len(df)} local files to hash.")
    
    local_paths = df['local_path'].tolist()
    path_map = dict(zip(df['local_path'], df['original_path']))
    num_workers = workers if workers else max(1, cpu_count() - 2)
    
    print(f"Hashing local copies using {num_workers} parallel workers...")
    exact_duplicates = defaultdict(list)
    
    with Pool(processes=num_workers) as pool:
        for local_path, file_hash in tqdm(pool.imap_unordered(process_file_hash, local_paths), total=len(local_paths)):
            if file_hash:
                orig_path = path_map.get(local_path)
                if orig_path:
                    exact_duplicates[file_hash].append(orig_path)
                
    # Filter out hashes that only have 1 file (meaning the size matched but content differed)
    confirmed_dupes = {h: paths for h, paths in exact_duplicates.items() if len(paths) > 1}
    
    num_duplicate_groups = len(confirmed_dupes)
    num_redundant_files = sum(len(paths) - 1 for paths in confirmed_dupes.values())
    
    print(f"\n--- Results ---")
    print(f"Found {num_duplicate_groups} groups of exact duplicate photos.")
    print(f"Total redundant files that can be safely deleted: {num_redundant_files}")
    
    if num_duplicate_groups > 0:
        # Save results to CSV
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving duplicate report to {output_path}...")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Duplicate_Group_ID', 'MD5_Hash', 'File_Path'])
            for i, (md5_hash, paths) in enumerate(confirmed_dupes.items(), 1):
                for path in paths:
                    writer.writerow([f"Group_{i:04d}", md5_hash, path])
                    
        print("Done! You can review the output CSV for files to safely delete on your external drive.")
    else:
        print("No exact duplicates found after hashing.")

    if cleanup:
        print("\nCleaning up local copies and mapping file...")
        for local_path in local_paths:
            try:
                os.remove(local_path)
            except OSError:
                pass
                
        try:
            os.remove(mapping_csv)
            print("Cleanup complete.")
        except OSError:
            pass

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    
    DEFAULT_MAPPING = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "copy_mapping.csv"
    DEFAULT_OUTPUT = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "exact_duplicates.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=str, default=str(DEFAULT_MAPPING), help="Path to mapping CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save duplicate report")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep local copies after hashing")
    args = parser.parse_args()
    
    main(args.mapping, args.output, args.workers, not args.no_cleanup)
