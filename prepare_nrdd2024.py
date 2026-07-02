"""
N-RDD2024 Dataset Preparation Script
=====================================
Run this LOCALLY before uploading to Kaggle.

What it does:
  1. Scans your downloaded N-RDD2024 folder
  2. Extracts all nested zip files automatically
  3. Finds all images and labels across every subfolder
  4. Prints the complete folder tree so you know exactly what to upload
  5. Optionally reorganises everything into a clean train/valid/test structure

Usage:
  python prepare_nrdd2024.py --root "E:/N-RDD2024"
  python prepare_nrdd2024.py --root "E:/N-RDD2024" --reorganise --output "E:/N-RDD2024-Clean"

Requirements:
  pip install tqdm
"""

import os
import sys
import shutil
import zipfile
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── N-RDD2024 class definitions ───────────────────────────────────────────────
CLASSES = {
    'D00': 'Longitudinal crack',
    'D10': 'Transverse crack',
    'D20': 'Alligator crack',
    'D30': 'Repaired crack',
    'D40': 'Pothole',
    'D50': 'Pedestrian crossing blur',
    'D60': 'Lane line blur',
    'D70': 'Manhole cover',
    'D80': 'Patchy road',
    'D90': 'Rutting',
}

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LBL_EXT = {'.txt', '.xml', '.json'}


# ── Step 1: Extract all zips recursively ────────────────────────────────────

def extract_all_zips(root: Path, verbose: bool = True) -> list[Path]:
    """Find and extract every zip file under root, including nested zips."""
    extracted = []
    round_num  = 0

    while True:
        zips = list(root.rglob('*.zip'))
        if not zips:
            break
        round_num += 1
        if verbose:
            print(f"\nRound {round_num}: found {len(zips)} zip file(s) to extract...")

        for zpath in tqdm(zips, desc=f'Extracting round {round_num}', disable=not verbose):
            out_dir = zpath.parent / zpath.stem
            try:
                with zipfile.ZipFile(zpath, 'r') as z:
                    z.extractall(out_dir)
                extracted.append(zpath)
                # Remove original zip to avoid re-extracting
                zpath.unlink()
            except Exception as e:
                print(f"  WARNING: Could not extract {zpath.name}: {e}")

    if verbose:
        print(f"\nTotal zip files extracted: {len(extracted)}")
    return extracted


# ── Step 2: Scan all images and labels ───────────────────────────────────────

def scan_dataset(root: Path) -> dict:
    """
    Walk root and collect all images and label files.
    Returns summary dict.
    """
    images = []
    labels = []
    countries = defaultdict(int)

    # Country folder patterns in N-RDD2024
    country_keywords = {
        'India': ['india', 'IND', 'ind'],
        'Japan': ['japan', 'JPN', 'jpn'],
        'Czech': ['czech', 'CZE', 'cze'],
        'Norway': ['norway', 'NOR', 'nor'],
        'China': ['china', 'CHN', 'chn'],
        'USA': ['usa', 'US', 'united'],
    }

    for p in root.rglob('*'):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMG_EXT:
                images.append(p)
                # Try to identify country from path
                path_str = str(p).lower()
                for country, keywords in country_keywords.items():
                    if any(k.lower() in path_str for k in keywords):
                        countries[country] += 1
                        break
                else:
                    countries['Unknown'] += 1
            elif ext in LBL_EXT:
                labels.append(p)

    return {
        'images' : images,
        'labels' : labels,
        'countries': dict(countries),
    }


# ── Step 3: Print full folder tree ───────────────────────────────────────────

def print_tree(root: Path, max_depth: int = 4, max_files: int = 5):
    """Print a concise directory tree."""
    print(f"\n{'='*60}")
    print(f"FOLDER TREE: {root}")
    print('='*60)

    def _walk(path: Path, depth: int, prefix: str):
        if depth > max_depth:
            return
        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return

        dirs  = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        for d in dirs:
            print(f"{prefix}📁 {d.name}/")
            _walk(d, depth + 1, prefix + '    ')

        show_files = files[:max_files]
        for f in show_files:
            size = f.stat().st_size / 1024
            print(f"{prefix}📄 {f.name}  ({size:.1f} KB)")
        if len(files) > max_files:
            print(f"{prefix}    ... and {len(files) - max_files} more files")

    _walk(root, 0, '')


# ── Step 4: Detect annotation format ─────────────────────────────────────────

def detect_format(labels: list[Path]) -> str:
    """Detect YOLO vs Pascal VOC vs COCO format."""
    if not labels:
        return 'unknown'
    # Sample first label
    sample = labels[0]
    ext = sample.suffix.lower()
    if ext == '.txt':
        try:
            content = sample.read_text().strip()
            if content:
                parts = content.split('\n')[0].split()
                if len(parts) == 5 and all(
                    p.replace('.','').replace('-','').isdigit()
                    for p in parts
                ):
                    return 'yolo'
        except Exception:
            pass
        return 'txt-unknown'
    elif ext == '.xml':
        return 'pascal_voc'
    elif ext == '.json':
        return 'coco'
    return 'unknown'


# ── Step 5: Parse YOLO labels to count class distribution ────────────────────

def count_class_distribution(labels: list[Path]) -> dict[int, int]:
    """Count occurrences of each class id across all YOLO label files."""
    counts = defaultdict(int)
    for lbl_path in tqdm(labels, desc='Counting classes', leave=False):
        try:
            for line in lbl_path.read_text().strip().split('\n'):
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
        except Exception:
            pass
    return dict(sorted(counts.items()))


# ── Step 6: Reorganise into clean train/valid/test split ─────────────────────

def reorganise(
    images: list[Path],
    labels: list[Path],
    output_dir: Path,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Reorganise all images+labels into:
    output_dir/
        train/images/  train/labels/
        valid/images/  valid/labels/
        test/images/   test/labels/
        data.yaml
    """
    random.seed(seed)

    # Match images to their labels (by stem)
    label_map = {lbl.stem: lbl for lbl in labels}
    paired    = []
    no_label  = []

    for img in images:
        lbl = label_map.get(img.stem)
        if lbl:
            paired.append((img, lbl))
        else:
            no_label.append(img)

    print(f"\nPaired  image+label : {len(paired)}")
    print(f"Images without label: {len(no_label)} (skipped)")

    random.shuffle(paired)
    n       = len(paired)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        'train': paired[:n_train],
        'valid': paired[n_train:n_train + n_valid],
        'test' : paired[n_train + n_valid:],
    }

    for split, pairs in splits.items():
        img_out = output_dir / split / 'images'
        lbl_out = output_dir / split / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(pairs, desc=f'Copying {split}'):
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"  {split:6s}: {len(pairs)} pairs")

    # Write data.yaml
    class_names = list(CLASSES.values())
    yaml_lines  = [
        f"path: {output_dir}",
        f"train: train/images",
        f"val:   valid/images",
        f"test:  test/images",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
        "# Class codes:",
    ] + [f"#   {code}: {name}" for code, name in CLASSES.items()]

    yaml_path = output_dir / 'data.yaml'
    yaml_path.write_text('\n'.join(yaml_lines))
    print(f"\ndata.yaml written → {yaml_path}")

    return splits


# ── Step 7: Print Kaggle upload instructions ──────────────────────────────────

def print_kaggle_instructions(root: Path, organised: bool, output_dir: Path = None):
    upload_path = output_dir if organised and output_dir else root
    print(f"""
{'='*60}
KAGGLE UPLOAD INSTRUCTIONS
{'='*60}

Folder to upload to Kaggle:
  {upload_path}

Steps:
  1. Go to kaggle.com → Datasets → New Dataset
  2. Dataset title: N-RDD2024-BD
  3. Upload the folder above (drag & drop or zip it first)
  4. Set licence: CC BY 4.0
  5. Click Create

In your Kaggle notebook:
  import os
  DATASET_ROOT = Path('/kaggle/input/n-rdd2024-bd')

  # If you used --reorganise:
  DATA_YAML = '/kaggle/input/n-rdd2024-bd/data.yaml'

  # Verify:
  for split in ['train', 'valid', 'test']:
      p = DATASET_ROOT / split / 'images'
      imgs = list(p.glob('*.jpg'))
      print(f"{{split}}: {{len(imgs)}} images")

Tip: If 6 GB is too large to upload in one go, upload
  train/ and valid/ as one dataset, test/ as a second dataset.
{'='*60}
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='N-RDD2024 dataset preparation for Kaggle')
    parser.add_argument('--root', type=str, required=True,
                        help='Root folder of downloaded N-RDD2024 dataset')
    parser.add_argument('--reorganise', action='store_true',
                        help='Reorganise into clean train/valid/test structure')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder for reorganised dataset '
                             '(default: <root>-Clean)')
    parser.add_argument('--train-ratio', type=float, default=0.80)
    parser.add_argument('--valid-ratio', type=float, default=0.10)
    parser.add_argument('--no-extract', action='store_true',
                        help='Skip zip extraction (if already extracted)')
    parser.add_argument('--tree-depth', type=int, default=4,
                        help='Depth of printed folder tree (default: 4)')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Root folder not found: {root}")
        sys.exit(1)

    print(f"\nN-RDD2024 Dataset Preparation")
    print(f"Root: {root}")
    print('='*60)

    # ── 1. Extract zips ───────────────────────────────────────────────────────
    if not args.no_extract:
        print("\nSTEP 1: Extracting zip files...")
        extract_all_zips(root)
    else:
        print("\nSTEP 1: Skipping extraction (--no-extract)")

    # ── 2. Print folder tree ──────────────────────────────────────────────────
    print("\nSTEP 2: Folder structure after extraction")
    print_tree(root, max_depth=args.tree_depth)

    # ── 3. Scan dataset ───────────────────────────────────────────────────────
    print("\nSTEP 3: Scanning all images and labels...")
    scan = scan_dataset(root)

    images = scan['images']
    labels = scan['labels']

    print(f"\n  Total images : {len(images):,}")
    print(f"  Total labels : {len(labels):,}")

    if scan['countries']:
        print("\n  Images by country (estimated from folder names):")
        for country, count in sorted(scan['countries'].items(),
                                     key=lambda x: -x[1]):
            print(f"    {country:10s}: {count:,}")

    # ── 4. Detect format ──────────────────────────────────────────────────────
    fmt = detect_format(labels)
    print(f"\n  Annotation format detected: {fmt}")
    if fmt != 'yolo':
        print("  WARNING: Non-YOLO format detected.")
        print("  You may need to convert annotations before training.")
        print("  Tool: https://github.com/ultralytics/JSON2YOLO")

    # ── 5. Class distribution ─────────────────────────────────────────────────
    if fmt == 'yolo' and labels:
        print("\nSTEP 4: Counting class distribution (this may take a moment)...")
        dist = count_class_distribution(labels[:5000])  # sample first 5000
        print("\n  Class distribution (sampled from first 5,000 label files):")
        print(f"  {'ID':>4}  {'Code':>6}  {'Name':30}  {'Count':>8}")
        print("  " + "-"*55)

        # N-RDD2024 class IDs 0-9 → D00-D90
        code_list = list(CLASSES.keys())
        name_list = list(CLASSES.values())
        for cls_id, count in sorted(dist.items()):
            code = code_list[cls_id] if cls_id < len(code_list) else f'cls{cls_id}'
            name = name_list[cls_id] if cls_id < len(name_list) else 'Unknown'
            print(f"  {cls_id:>4}  {code:>6}  {name:30}  {count:>8,}")

    # ── 6. Reorganise ─────────────────────────────────────────────────────────
    output_dir = None
    if args.reorganise:
        output_dir = Path(args.output) if args.output else Path(str(root) + '-Clean')
        print(f"\nSTEP 5: Reorganising into {output_dir} ...")
        if output_dir.exists():
            print(f"  Output folder already exists. Contents will be overwritten.")
        output_dir.mkdir(parents=True, exist_ok=True)

        reorganise(
            images, labels, output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )
        print(f"\nClean dataset saved → {output_dir}")
    else:
        print("\nSTEP 5: Skipping reorganisation (pass --reorganise to enable)")
        print("  Upload the extracted root folder directly to Kaggle.")

    # ── 7. Final summary ──────────────────────────────────────────────────────
    print(f"""
{'='*60}
SUMMARY
{'='*60}
  Images found : {len(images):,}
  Labels found : {len(labels):,}
  Format       : {fmt}
  Classes      : {len(CLASSES)} (D00–D90)
  Reorganised  : {'Yes → ' + str(output_dir) if output_dir else 'No'}
""")

    print_kaggle_instructions(root, args.reorganise, output_dir)


if __name__ == '__main__':
    main()
