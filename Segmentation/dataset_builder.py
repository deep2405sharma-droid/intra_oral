"""
dataset_builder.py
──────────────────
Builds three dataset CSV files for Mask R-CNN training from
smart_merged_with_coco.csv.

Dataset configurations:
    Dataset A — smart_II (normal + opmd + variation)
                + smart_om (opmd + variation only)

    Dataset B — smart_II (opmd + variation only)
                + smart_om (normal + opmd + variation)

    Dataset C — Both sources, all three classes
                (normal + opmd + variation from smart_II + smart_om)

    Dataset D — Both sources, all three classes
                BUT only 30% of normal images (partial normals)
                to reduce class imbalance while keeping some normals

Each output CSV contains only rows with:
    - valid coco_file (not null)
    - image_path exists on disk
    - coco_file exists on disk
    - lesion_location in valid 7 locations (excludes LA, UB, RB1/2/3)

Usage:
    python -m Segmentation.dataset_builder

Output:
    data/coco/dataset_A.csv
    data/coco/dataset_B.csv
    data/coco/dataset_C.csv
    data/coco/dataset_D.csv
"""

import os
import configparser
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
CONFIG_PATH = r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\Segmentation\config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

coco_output_dir = config.get("COCO", "coco.output.dir")

# Source CSV — output of prepare_coco_json.py (has coco_file column)
CSV_PATH = os.path.join(coco_output_dir, "smart_merged_with_coco.csv")

# Valid lesion locations (exclude LA, UB, RB1/RB2/RB3)
VALID_LOCATIONS = {"DT", "LB", "RB", "UL", "UA", "VT", "LL"}

# Location remap
LOCATION_REMAP = {"RB1": "RB", "RB2": "RB", "RB3": "RB"}


# ── Helper functions ──────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, normalize lesion_location, keep only rows with
    valid coco_file and files that exist on disk.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows")

    # Normalize lesion_location
    df["lesion_location"] = df["lesion_location"].apply(
        lambda x: LOCATION_REMAP.get(str(x).strip(), str(x).strip())
    )

    # Keep only valid locations
    df = df[df["lesion_location"].isin(VALID_LOCATIONS)].copy()
    print(f"After location filter: {len(df)} rows")

    # Keep only rows with coco_file
    df = df[df["coco_file"].notna()].copy()
    print(f"After coco_file filter: {len(df)} rows")

    # Keep only rows where both files exist on disk
    exists_mask = (
        df["image_path"].apply(lambda p: os.path.exists(str(p))) &
        df["coco_file"].apply(lambda p: os.path.exists(str(p)))
    )
    df = df[exists_mask].reset_index(drop=True)
    print(f"After disk check: {len(df)} rows")

    return df


def build_dataset(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Filter DataFrame based on rules dict:
        rules = {
            "smart_II": ["normal", "opmd", "variation"],
            "smart_om": ["opmd", "variation"],
        }
    """
    parts = []
    for source, labels in rules.items():
        part = df[
            (df["source"] == source) &
            (df["label"].isin(labels))
        ].copy()
        parts.append(part)
        print(f"  {source} ({', '.join(labels)}): {len(part)} rows")

    result = pd.concat(parts, ignore_index=True)
    return result


def build_dataset_partial_normal(
    df: pd.DataFrame,
    rules: dict,
    normal_fraction: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Like build_dataset but takes only 30% of normal patient IDs
    from each source, keeping all opmd and variation rows.

    Sampling is done by patient_id to avoid partial patient data.

    Args:
        rules          : same format as build_dataset
        normal_fraction: fraction of normal patient IDs to keep (default 0.3 = 30%)
        seed           : random seed for reproducibility
    """
    parts = []
    for source, labels in rules.items():
        non_normal_labels = [l for l in labels if l != "normal"]
        normal_labels     = [l for l in labels if l == "normal"]

        # All opmd + variation rows
        if non_normal_labels:
            part_non_normal = df[
                (df["source"] == source) &
                (df["label"].isin(non_normal_labels))
            ].copy()
            parts.append(part_non_normal)
            print(f"  {source} ({', '.join(non_normal_labels)}): {len(part_non_normal)} rows")

        # 30% of normal patient IDs
        if normal_labels:
            normal_rows = df[
                (df["source"] == source) &
                (df["label"] == "normal")
            ].copy()

            import random
            all_pids     = normal_rows["patient_id"].unique()
            n_sample     = max(1, int(len(all_pids) * normal_fraction))
            random.seed(seed)
            sampled_pids = random.sample(sorted(all_pids.tolist()), n_sample)

            part_normal = normal_rows[normal_rows["patient_id"].isin(sampled_pids)].copy()
            parts.append(part_normal)
            print(f"  {source} (normal 30%): "
                  f"{n_sample}/{len(all_pids)} patients → {len(part_normal)} rows")

    result = pd.concat(parts, ignore_index=True)
    return result
    """
    Filter DataFrame based on rules dict:
        rules = {
            "smart_II": ["normal", "opmd", "variation"],
            "smart_om": ["opmd", "variation"],
        }
    """
    parts = []
    for source, labels in rules.items():
        part = df[
            (df["source"] == source) &
            (df["label"].isin(labels))
        ].copy()
        parts.append(part)
        print(f"  {source} ({', '.join(labels)}): {len(part)} rows")

    result = pd.concat(parts, ignore_index=True)
    return result


def print_summary(name: str, df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total rows : {len(df)}")
    print(f"  By source + label:")
    summary = df.groupby(["source", "label"]).size()
    for (src, lbl), cnt in summary.items():
        print(f"    {src:<12} {lbl:<12} : {cnt}")
    print(f"  By lesion_location:")
    loc_summary = df["lesion_location"].value_counts()
    for loc, cnt in loc_summary.items():
        print(f"    {loc:<6} : {cnt}")
    print(f"{'='*55}")


def save_dataset(df: pd.DataFrame, name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Loading from: {CSV_PATH}")
    df = load_and_clean(CSV_PATH)

    print(f"\nFull cleaned dataset: {len(df)} rows")
    print(df.groupby(["source", "label"]).size().to_string())

    # ── Dataset A ─────────────────────────────────────────────────
    # smart_II: normal + opmd + variation
    # smart_om: opmd + variation ONLY
    print("\nBuilding Dataset A...")
    rules_A = {
        "smart_II": ["normal", "opmd", "variation"],
        "smart_om": ["opmd", "variation"],
    }
    df_A = build_dataset(df, rules_A)
    print_summary("Dataset A", df_A)
    save_dataset(df_A, "dataset_A", coco_output_dir)

    # ── Dataset B ─────────────────────────────────────────────────
    # smart_II: opmd + variation ONLY
    # smart_om: normal + opmd + variation
    print("\nBuilding Dataset B...")
    rules_B = {
        "smart_II": ["opmd", "variation"],
        "smart_om": ["normal", "opmd", "variation"],
    }
    df_B = build_dataset(df, rules_B)
    print_summary("Dataset B", df_B)
    save_dataset(df_B, "dataset_B", coco_output_dir)

    # ── Dataset C ─────────────────────────────────────────────────
    # Both sources, all three classes
    print("\nBuilding Dataset C...")
    rules_C = {
        "smart_II": ["normal", "opmd", "variation"],
        "smart_om": ["normal", "opmd", "variation"],
    }
    df_C = build_dataset(df, rules_C)
    print_summary("Dataset C", df_C)
    save_dataset(df_C, "dataset_C", coco_output_dir)

    # ── Dataset D ─────────────────────────────────────────────────
    # Both sources, all three classes
    # BUT only partial (30%) normal images from each source
    # Useful to reduce class imbalance while keeping some normals
    print("\nBuilding Dataset D (partial normals)...")
    rules_D = {
        "smart_II": ["normal", "opmd", "variation"],
        "smart_om": ["normal", "opmd", "variation"],
    }
    df_D = build_dataset_partial_normal(
        df,
        rules=rules_D,
        normal_fraction=0.3,   # keep 30% of normal images — change as needed
        seed=42,
    )
    print_summary("Dataset D", df_D)
    save_dataset(df_D, "dataset_D", coco_output_dir)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  All datasets saved to: {coco_output_dir}")
    print(f"  Dataset A : {len(df_A)} rows")
    print(f"  Dataset B : {len(df_B)} rows")
    print(f"  Dataset C : {len(df_C)} rows")
    print(f"  Dataset D : {len(df_D)} rows  (30% normals)")
    print(f"{'='*55}")
    print("\nUpdate maskrcnn.ini:")
    print(f"  csv.path = {coco_output_dir}\\dataset_A.csv   # or B, C, D")