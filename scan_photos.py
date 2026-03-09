r"""
scan_photos.py  — Scan an external drive for photos and produce a detailed CSV catalog.

Walks the configured folders on the drive, collects file-system metadata and
EXIF data via exifread, flattens key fields into top-level columns, and writes
everything to a single CSV in the Bronze data lake.

Usage
-----
    python scan_photos.py                         # defaults: E:\ → Bronze
    python scan_photos.py --root F:\              # different drive
    python scan_photos.py --output ./catalog.csv  # custom output path
"""

import argparse
import csv
import datetime
import multiprocessing
import os
import sys
from pathlib import Path
import exifread


# ── Configuration ────────────────────────────────────────────────
DEFAULT_ROOT = "E:\\"
DEFAULT_FOLDERS = (
    "1_3rd_Year_Photos",
    "2_4th_Year_Photos",
    "3_5th_Year_Photos",
    "4_Post_Grad",
    "5_DeltStuff",
    "6_Grad_School",
    "7_IPhone_Photos",
    "8_Life",
)
FILE_TYPES = {".jpg", ".jpeg", ".nef", ".arw", ".png", ".tiff", ".tif"}

# EXIF fields to extract as top-level CSV columns.
# Keys are the exifread tag names; values are the CSV column names.
EXIF_FIELDS = {
    "Image Make":               "camera_make",
    "Image Model":              "camera_model",
    "Image Orientation":        "exif_orientation",
    "EXIF LensModel":           "lens_model",
    "EXIF ExposureTime":        "exposure_time",
    "EXIF FNumber":             "f_number",
    "EXIF ISOSpeedRatings":     "iso",
    "Image DateTime":           "image_datetime",
    "EXIF DateTimeOriginal":    "datetime_original",
    "EXIF DateTimeDigitized":   "datetime_digitized",
    "EXIF ExifImageWidth":      "image_width",
    "EXIF ExifImageLength":     "image_height",
    "EXIF ColorSpace":          "color_space",
    "Interop InteropIndex":     "interop_index",
    "EXIF FocalLength":         "focal_length",
    "EXIF Flash":               "flash",
    "GPS GPSLatitude":          "gps_latitude",
    "GPS GPSLatitudeRef":       "gps_latitude_ref",
    "GPS GPSLongitude":         "gps_longitude",
    "GPS GPSLongitudeRef":      "gps_longitude_ref",
    "EXIF WhiteBalance":        "white_balance",
    "EXIF MeteringMode":        "metering_mode",
    "EXIF ExposureProgram":     "exposure_program",
    "EXIF ExposureBiasValue":   "exposure_bias",
}

# EXIF Orientation tag values → human-readable label + CW rotation degrees.
# These tell you how to transform raw pixels into correct display orientation.
#   1 = Normal                          →   0°
#   2 = Flipped horizontally            →   0° + horizontal flip
#   3 = Rotated 180°                    → 180°
#   4 = Flipped vertically              → 180° + horizontal flip
#   5 = Rotated 90° CCW + horiz flip    → 270° + horizontal flip
#   6 = Rotated 90° CW                  →  90°
#   7 = Rotated 90° CW + horiz flip     →  90° + horizontal flip
#   8 = Rotated 90° CCW                 → 270°
ORIENTATION_MAP = {
    "1": ("Normal",                  0,   False),
    "2": ("Flipped Horizontal",      0,   True),
    "3": ("Rotated 180",             180, False),
    "4": ("Flipped Vertical",        180, True),
    "5": ("Rotated 90 CCW + Flip",   270, True),
    "6": ("Rotated 90 CW",           90,  False),
    "7": ("Rotated 90 CW + Flip",    90,  True),
    "8": ("Rotated 90 CCW",          270, False),
}

# Silver data-lake path (relative to this script's repo root)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = REPO_ROOT / ".data_lake" / "01_Bronze" / "photo" / "photo_catalog.csv"


# ── Helpers ──────────────────────────────────────────────────────
def collect_file_data(file: Path) -> dict:
    """Return a dict of file-system metadata for *file*."""
    stat = file.stat()
    return {
        "location":     str(file.resolve()),
        "folder_name":  str(file.parent),
        "file_name":    file.name,
        "file_type":    file.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "dt_accessed":  datetime.datetime.fromtimestamp(stat.st_atime).isoformat(),
        "dt_modified":  datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "dt_created":   datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }


def extract_exif(file_path: str) -> dict:
    """Read EXIF tags and return a flat dict with the configured fields."""
    result = {col: "" for col in EXIF_FIELDS.values()}
    result["orientation_label"] = ""
    result["rotation_needed"] = ""
    result["flip_needed"] = ""
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        for tag_key, col_name in EXIF_FIELDS.items():
            val = tags.get(tag_key)
            if val is not None and not isinstance(val, bytes):
                result[col_name] = str(val).strip()

        # Derive orientation fields
        raw_orient = result.get("exif_orientation", "")
        if raw_orient in ORIENTATION_MAP:
            label, degrees, flip = ORIENTATION_MAP[raw_orient]
            result["orientation_label"] = label
            result["rotation_needed"] = degrees
            result["flip_needed"] = flip
    except Exception:
        pass  # Silently skip unreadable files
    return result


def discover_files(root: Path, folders: tuple) -> list:
    """Walk *root* / *folders* and return a list of Path objects for matching files."""
    files = []
    for folder_name in folders:
        folder_path = root / folder_name
        if not folder_path.is_dir():
            print(f"  ⚠  Folder not found, skipping: {folder_path}")
            continue
        print(f"  📂 Scanning: {folder_name}")
        for f in folder_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in FILE_TYPES:
                files.append(f)
    return files


def save_csv(path: Path, data: list) -> None:
    """Write *data* (list of dicts) to a CSV at *path*."""
    if not data:
        print("No data to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(data[0].keys())
    with open(path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def print_summary(data: list) -> None:
    """Print a readable summary of the scanned catalog."""
    total = len(data)
    print(f"\n{'='*60}")
    print(f"  Photo Catalog Summary")
    print(f"{'='*60}")
    print(f"  Total files found: {total:,}")

    # Breakdown by file type
    type_counts = {}
    for row in data:
        ft = row.get("file_type", "?")
        type_counts[ft] = type_counts.get(ft, 0) + 1
    print(f"\n  By file type:")
    for ft, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ft:>6}  {cnt:>6,}")

    # Breakdown by camera model
    model_counts = {}
    for row in data:
        model = row.get("camera_model", "") or "Unknown"
        model_counts[model] = model_counts.get(model, 0) + 1
    print(f"\n  By camera model:")
    for model, cnt in sorted(model_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {model:<30} {cnt:>6,}")

    # Total size
    total_mb = sum(row.get("file_size_mb", 0) for row in data)
    print(f"\n  Total size: {total_mb:,.1f} MB ({total_mb/1024:.1f} GB)")
    print(f"{'='*60}\n")


def process_file(file_path: Path) -> dict:
    """Process a single file: collect file data + EXIF. Used by multiprocessing."""
    row = collect_file_data(file_path)
    exif = extract_exif(row["location"])
    row.update(exif)
    return row


# ── Main ─────────────────────────────────────────────────────────
def main():
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(description="Scan photos and build a metadata catalog.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT,
                        help=f"Root path of the external drive (default: {DEFAULT_ROOT})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--all-folders", action="store_true",
                        help="Scan ALL top-level folders on the drive, not just the default 8")
    parser.add_argument("--workers", type=int, default=8,
                        help=f"Number of parallel workers (default: 8)")
    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.output)
    num_workers = max(1, args.workers)

    if not root.exists():
        print(f"Error: Root path does not exist: {root}")
        sys.exit(1)

    # Determine folders to scan
    if args.all_folders:
        folders = tuple(d.name for d in root.iterdir() if d.is_dir())
    else:
        folders = DEFAULT_FOLDERS

    print(f"Scanning {root} for photos in {len(folders)} folders...")
    files = discover_files(root, folders)
    total = len(files)
    print(f"\nFound {total:,} photo files. Extracting metadata with {num_workers} workers...\n")

    if not files:
        print("No files found. Exiting.")
        sys.exit(0)

    # Process files in parallel
    data = []
    chunk_size = max(1, total // (num_workers * 4))  # Balance granularity
    with multiprocessing.Pool(processes=num_workers) as pool:
        for idx, row in enumerate(pool.imap_unordered(process_file, files, chunksize=chunk_size), 1):
            data.append(row)
            if (idx % 500) == 0:
                print(f"  Processed {idx:,} / {total:,} files...")

    print(f"  Processed {total:,} / {total:,} files. Done.\n")

    # Summary
    print_summary(data)

    # Save
    save_csv(output_path, data)
    print(f"Saved catalog to: {output_path}")


if __name__ == "__main__":
    main()
