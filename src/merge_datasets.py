"""
merge_datasets.py
─────────────────
Concatenates the cleaned SMART-II and SMART-OM metadata CSVs into a single
unified dataset for joint OPMD / Variation modelling.

Schema after inspection of both CSVs:

  Common columns (37):
    patient_id, serial_no, age, sex, habit_history, habit_types, tobacco_type,
    smoking_*, chewing_*, arecanut_freq/duration/status/onset,
    oral_hygiene_*, family_history, past_medical/dental_history, denture_usage,
    label, lesion_location, lesion_present, lesion_classification,
    image_path, source, type

  SMART-II only (28):
    religion, marital_status, education, occupation, diet, spice_consumption,
    arecanut_type, arecanut_product_name, arecanut_quantity_sachets_per_day,
    arecanut_other, chewing_quit_age,
    oral_hygiene_other_aids, oral_hygiene_other_aids_details,
    denture_duration_years, denture_type, denture_material, denture_hurts,
    lesion_present_raw, lesion_colour, lesion_margin, lesion_surface_feature,
    lesion_description, lesion_size_cm, lesion_pain, lesion_pain_duration_days,
    lesion_pain_type, lesion_num_sites, other_findings

  SMART-OM only (4):
    alcohol_freq_per_week, alcohol_duration_years, alcohol_status,
    alcohol_onset_age

  NOTE — lesion_present bug in SMART-II CSV:
    The clean_smart_metadata.py expand_to_image_rows() derives lesion_present
    from the label column AFTER assign_label() runs, but at that point the
    label is already lowercased ('opmd'/'variation') while the check uses
    title-case ('OPMD'/'Variation').  Result: every SMART-II row has
    lesion_present=False.  This script corrects it during merge using
    lesion_present_raw (which correctly holds 'Present'/'Absent').

Usage:
    python merge_datasets.py

Config keys used:
    [SMART-II-DATAPATH]  smart.patient.clean.metadata.filename
    [SMART-OM-DATAPATH]  smartom.patient.clean.metadata.filename
    [MERGED]             merged.output.filename
                         merged.missing.filename
"""

import os
import numpy as np
import pandas as pd

from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config
from utils.log_handler import CustomSizeDayRotatingFileHandler


# ─────────────────────────────────────────────────────────────────────────────
# Column order for the merged output
# ─────────────────────────────────────────────────────────────────────────────

MERGED_COL_ORDER = [
    # Identity
    "patient_id",
    "label",
    "source",
    "type",
    "serial_no",
    # Image
    "image_path",
    "lesion_location",
    "lesion_present",
    "lesion_classification",
    # Demographics — shared
    "age",
    "sex",
    # Demographics — SMART-II only
    "religion",
    "marital_status",
    "education",
    "occupation",
    "diet",
    "spice_consumption",
    # Habits
    "habit_history",
    "habit_types",
    "tobacco_type",
    # Smoking — shared
    "smoking_type",
    "smoking_freq_per_day",
    "smoking_duration_years",
    "smoking_status",
    "smoking_onset_age",
    # Chewing — shared (SMART-II smokeless tobacco renamed to chewing_*)
    "chewing_freq_per_day",
    "chewing_duration_years",
    "chewing_habit_status",
    "chewing_onset_age",
    # Chewing — SMART-II only
    "chewing_quit_age",
    # Arecanut — shared
    "arecanut_freq_per_day",
    "arecanut_duration_years",
    "arecanut_status",
    "arecanut_onset_age",
    # Arecanut — SMART-II only
    "arecanut_type",
    "arecanut_product_name",
    "arecanut_quantity_sachets_per_day",
    "arecanut_other",
    # Alcohol — SMART-OM only
    "alcohol_freq_per_week",
    "alcohol_duration_years",
    "alcohol_status",
    "alcohol_onset_age",
    # Oral hygiene — shared
    "oral_hygiene_cleaning_aid",
    "oral_hygiene_material",
    "oral_hygiene_brushing_method",
    "oral_hygiene_brushing_freq",
    "oral_hygiene_brushing_duration",
    "oral_hygiene_brush_change_freq",
    # Oral hygiene — SMART-II only
    "oral_hygiene_other_aids",
    "oral_hygiene_other_aids_details",
    # Medical / dental — shared
    "family_history",
    "past_medical_history",
    "past_dental_history",
    "denture_usage",
    # Denture detail — SMART-II only
    "denture_duration_years",
    "denture_type",
    "denture_material",
    "denture_hurts",
    # Lesion detail — shared
    "lesion_classification",
    # Lesion detail — SMART-II only
    "lesion_colour",
    "lesion_margin",
    "lesion_surface_feature",
    "lesion_description",
    "lesion_size_cm",
    "lesion_pain",
    "lesion_num_sites",
    "lesion_pain_duration_days",
    "lesion_pain_type",
    "other_findings",
]


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def fix_lesion_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute lesion_present correctly from label (already lowercase).
    Overrides the broken value produced by clean_smart_metadata.py for
    SMART-II rows where the title-case check fired before lowercasing.
    For SMART-OM rows the value was already correct; recomputing is harmless.
    """
    df["lesion_present"] = df["label"].isin(["opmd", "variation"])
    return df


def normalise_lesion_present(series: pd.Series) -> pd.Series:
    """Coerce string 'True'/'False' from CSV round-trip to Python bool."""
    return series.map(
        lambda x: (
            True
            if str(x).strip().lower() == "true"
            else False if str(x).strip().lower() == "false" else np.nan
        )
    )


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = [
        {
            "column": col,
            "missing_n": int(df[col].isna().sum()),
            "missing_pct": (
                round(df[col].isna().sum() / total * 100, 1) if total else 0.0
            ),
        }
        for col in df.columns
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────


def merge_datasets(
    logger: CustomSizeDayRotatingFileHandler,
    smart2_csv: str,
    smartom_csv: str,
    output_csv: str,
    report_csv: str,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("  Merging SMART-II + SMART-OM datasets")
    logger.info("=" * 60)

    # Load
    df_s2 = pd.read_csv(smart2_csv, dtype=str)
    df_om = pd.read_csv(smartom_csv, dtype=str)
    logger.info(f"  SMART-II  rows: {len(df_s2):>6}  cols: {df_s2.shape[1]}")
    logger.info(f"  SMART-OM  rows: {len(df_om):>6}  cols: {df_om.shape[1]}")

    # Fix lesion_present (SMART-II bug + normalise both to bool)
    df_s2 = fix_lesion_present(df_s2)
    df_om = fix_lesion_present(df_om)
    logger.info("  lesion_present recomputed from label for both datasets.")

    # Concatenate (outer join fills dataset-exclusive cols with NaN)
    df_merged = pd.concat([df_s2, df_om], axis=0, join="outer", ignore_index=True)
    logger.info(f"  Merged    rows: {len(df_merged):>6}  cols: {df_merged.shape[1]}")

    # Drop lesion_present_raw — only needed for the fix above
    if "lesion_present_raw" in df_merged.columns:
        df_merged.drop(columns=["lesion_present_raw"], inplace=True)
        logger.info("  Dropped intermediate column: lesion_present_raw")

    # Column order
    seen, ordered = set(), []
    for c in MERGED_COL_ORDER:
        if c in df_merged.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    extras = [c for c in df_merged.columns if c not in seen]
    df_merged = df_merged[ordered + extras]

    # Log distributions
    logger.info("  Label distribution:")
    for lbl, cnt in df_merged["label"].value_counts().items():
        pct = 100 * cnt / len(df_merged)
        logger.info(f"    {lbl:12s}  {cnt:5d}  ({pct:5.1f}%)")

    logger.info("  Source distribution:")
    for src, cnt in df_merged["source"].value_counts().items():
        logger.info(f"    {src:12s}  {cnt:5d}")

    logger.info("  lesion_present distribution:")
    for val, cnt in df_merged["lesion_present"].value_counts().items():
        logger.info(f"    {str(val):8s}  {cnt:5d}")

    # Missingness report
    miss = missingness_report(df_merged)

    # Save
    os.makedirs(
        os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(report_csv) if os.path.dirname(report_csv) else ".",
        exist_ok=True,
    )
    df_merged.to_csv(output_csv, index=False)
    miss.to_csv(report_csv, index=False)

    logger.info(f"  Saved merged CSV     → {output_csv}")
    logger.info(f"  Saved missing report → {report_csv}")
    logger.info("=" * 60)

    return df_merged


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def initialize_logger(config):
    log_filename = config.get("LOGGER", "logger.filename")
    return iolog.getLogger(log_filename)


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config=config)

    smart2_csv = config.get(
        "SMART-II-DATAPATH", "smart.patient.clean.metadata.filename"
    )
    smartom_csv = config.get(
        "SMART-OM-DATAPATH", "smartom.patient.clean.metadata.filename"
    )
    output_csv = config.get("SMART_MERGED", "merged.output.filename")
    report_csv = config.get("SMART_MERGED", "merged.missing.filename")

    df = merge_datasets(
        logger=logger,
        smart2_csv=smart2_csv,
        smartom_csv=smartom_csv,
        output_csv=output_csv,
        report_csv=report_csv,
    )
