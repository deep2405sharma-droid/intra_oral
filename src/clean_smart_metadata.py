"""
clean_smart_metadata.py
────────────────────────
Cleans Patient-Metadata-SMART-SMITA.xlsx into a structured CSV with
ONE ROW PER IMAGE, so it can be directly concatenated with the SMART-OM
cleaned CSV for joint OPMD / Variation modelling.

Disk structure (SMART-II):
    <images_dir>/<label>/<patient_id>/Unannotated/<patient_id>_<visit>_<site>.jpg
    <images_dir>/<label>/<patient_id>/Json file/*.json
    <images_dir>/<label>/<patient_id>/Json/*.json       (alternate spelling)

Output row schema (aligned with SMART-OM clean CSV):
    patient_id, <all metadata cols>, label, lesion_location,
    lesion_present, lesion_classification, image_path, source, type
"""

import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config
from utils.log_handler import CustomSizeDayRotatingFileHandler


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SITE_CODES = [
    "LB",
    "RB",
    "DT",
    "VT",
    "UL",
    "LL",
    "UDA",
    "LDA",
    "UA",
    "LA",
    "RB2",
    "RB3",
    "UB",
]

OPMD_CLASSES = {"Homogenous Leukoplakia", "Lichen Planus"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
JSON_EXTENSION = ".json"

# Columns to rename (raw Excel name → clean snake_case name)
COL_RENAME = {
    "S.No": "serial_no",
    "SMITA ID": "patient_id",
    "Age in years": "age",
    "Sex": "sex",
    "Religion": "religion",
    "Marital status": "marital_status",
    "Education": "education",
    "Occupation": "occupation",
    "Diet history": "diet",
    "Spice consumption": "spice_consumption",
    "Habit history": "habit_history",
    "If yes, mention habits": "habit_types",
    "If Tobacco, mention the type": "tobacco_type",
    # Smoking
    "SMOKING FORM OF TOBACCO": "smoking_type",
    "Unnamed: 14": "smoking_freq_per_day",
    "Unnamed: 15": "smoking_duration_years",
    "Unnamed: 16": "smoking_status",
    "Unnamed: 17": "smoking_onset_age",
    "Unnamed: 18": "smoking_quit_age",
    # Smokeless tobacco
    "SMOKELESS FORM OF TOBACCO": "chewing_freq_per_day",
    "Unnamed: 20": "chewing_duration_years",
    "Unnamed: 21": "chewing_habit_status",
    "Unnamed: 22": "chewing_onset_age",
    "Unnamed: 23": "chewing_quit_age",
    # Arecanut
    "ARECANUT": "arecanut_type",
    "Unnamed: 25": "arecanut_product_name",
    "Unnamed: 26": "arecanut_freq_per_day",
    "Unnamed: 27": "arecanut_duration_years",
    "Unnamed: 28": "arecanut_quantity_sachets_per_day",
    "Unnamed: 29": "arecanut_status",
    "Unnamed: 30": "arecanut_onset_age",
    "Unnamed: 31": "arecanut_quit_age",
    "Unnamed: 32": "arecanut_other",
    # Alcohol
    "ALCOHOL": "alcohol_type",
    "Unnamed: 34": "alcohol_freq_per_week",
    "Unnamed: 35": "alcohol_duration_years",
    "Unnamed: 36": "alcohol_quantity_ml_per_week",
    "Unnamed: 37": "alcohol_status",
    "Unnamed: 38": "alcohol_onset_age",
    "Unnamed: 39": "alcohol_quit_age",
    # Oral hygiene
    "ORAL HYGIENE PRACTICES": "oral_hygiene_cleaning_aid",
    "Unnamed: 41": "oral_hygiene_material",
    "Unnamed: 42": "oral_hygiene_brushing_method",
    "Unnamed: 43": "oral_hygiene_brushing_freq",
    "Unnamed: 44": "oral_hygiene_brushing_duration",
    "Unnamed: 45": "oral_hygiene_brush_change_freq",
    "Unnamed: 46": "oral_hygiene_other_aids",
    "Unnamed: 47": "oral_hygiene_other_aids_details",
    # Medical history
    "Family history": "family_history",
    "Past Medical history": "past_medical_history",
    # Denture
    "Past Dental History": "past_dental_history",
    "DENTURE HISTORY": "denture_usage",
    "Unnamed: 52": "denture_duration_years",
    "Unnamed: 53": "denture_type",
    "Unnamed: 54": "denture_material",
    "Unnamed: 55": "denture_hurts",
    # Lesion
    "Presence or Absence of lesion": "lesion_present",
    "Location": "lesion_location",
    "Colour": "lesion_colour",
    "Margin": "lesion_margin",
    "Surface feature": "lesion_surface_feature",
    "Description of the Lesion": "lesion_description",
    "If multiple lesion present, mention number of sites involved": "lesion_num_sites",
    "Size in cms (Length x Width)": "lesion_size_cm",
    "Associated with pain": "lesion_pain",
    "If yes, duration of pain (in days)": "lesion_pain_duration_days",
    "Type of pain": "lesion_pain_type",
    "Lesion classification": "lesion_classification",
    "Others": "other_findings",
}

# Columns dropped because >85% missing — not clinically recoverable
# DROP_COLS = {
#     "serial_no",
#     "smoking_quit_age",
#     "smokeless_quit_age",
#     "arecanut_quit_age",
#     "arecanut_product_name",
#     "arecanut_quantity_sachets_per_day",
#     "arecanut_other",
#     "alcohol_type",
#     "alcohol_freq_per_week",
#     "alcohol_duration_years",
#     "alcohol_quantity_ml_per_week",
#     "alcohol_status",
#     "alcohol_onset_age",
#     "alcohol_quit_age",
#     "oral_hygiene_other_aids_details",
#     "denture_duration_years",
#     "denture_type",
#     "denture_material",
#     "denture_hurts",
#     "lesion_num_sites",
#     "lesion_pain_duration_days",
#     "lesion_pain_type",
#     # raw lesion_present replaced by boolean lesion_present column
#     "lesion_present",
# }

# Final column order for the one-row-per-image output
COL_ORDER = [
    "patient_id",
    "label",
    "age",
    "sex",
    "religion",
    "marital_status",
    "education",
    "occupation",
    "diet",
    "spice_consumption",
    "habit_history",
    "habit_types",
    "tobacco_type",
    "smoking_type",
    "smoking_freq_per_day",
    "smoking_duration_years",
    "smoking_status",
    "smoking_onset_age",
    "smokeless_freq_per_day",
    "smokeless_duration_years",
    "smokeless_status",
    "smokeless_onset_age",
    "arecanut_type",
    "arecanut_freq_per_day",
    "arecanut_duration_years",
    "arecanut_status",
    "arecanut_onset_age",
    "oral_hygiene_cleaning_aid",
    "oral_hygiene_material",
    "oral_hygiene_brushing_method",
    "oral_hygiene_brushing_freq",
    "oral_hygiene_brushing_duration",
    "oral_hygiene_brush_change_freq",
    "oral_hygiene_other_aids",
    "family_history",
    "past_medical_history",
    "past_dental_history",
    "denture_usage",
    "lesion_present",
    "lesion_location",
    "lesion_colour",
    "lesion_margin",
    "lesion_surface_feature",
    "lesion_description",
    "lesion_classification",
    "lesion_size_cm",
    "lesion_pain",
    "other_findings",
    "image_path",
    "source",
    "type",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────


def normalise_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "habit_history" in df.columns:
        df["habit_history"] = (
            df["habit_history"]
            .apply(lambda x: x.strip().capitalize() if isinstance(x, str) else x)
            .replace({"Yes": "Yes", "No": "No"})
        )
    if "lesion_present" in df.columns:
        df["lesion_present"] = df["lesion_present"].apply(
            lambda x: x.strip().capitalize() if isinstance(x, str) else x
        )
    if "sex" in df.columns:
        df["sex"] = df["sex"].apply(
            lambda x: x.strip().capitalize() if isinstance(x, str) else x
        )
    lc_map = {
        "Homogenous leukoplakia": "Homogenous Leukoplakia",
        "Homogemous leukoplakia": "Homogenous Leukoplakia",
        "Lichen Planus": "Lichen Planus",
        "Frictional keratosis": "Frictional Keratosis",
        "No OPMD": "No OPMD",
    }
    if "lesion_classification" in df.columns:
        df["lesion_classification"] = df["lesion_classification"].apply(
            lambda x: lc_map.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
        )
    return df


# def assign_label(row: pd.Series) -> str:
#     present = str(row.get("lesion_present", "")).strip().lower()
#     classif = str(row.get("lesion_classification", "")).strip()
#     if present == "absent":
#         return "normal"
#     elif present == "present":
#         return "opmd" if classif in OPMD_CLASSES else "variation"
#     return "Unknown"


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    miss_pct = (df.isna().sum() / len(df) * 100).round(1)
    miss_n = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_n": miss_n.values,
                "missing_pct": miss_pct.values,
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image collection — one record per image file
# ─────────────────────────────────────────────────────────────────────────────


def _site_code_from_filename(fname: str) -> str | None:
    """
    Extract the site code from the image filename stem.
    Convention: <patient_id>_<visit_code>_<SITE>[<digit>].jpg
    e.g. SMITA00402_R_LB.jpg  -> LB
         SMITA00402_R_RB2.jpg -> RB2
    Falls back to the last underscore-segment (stripped of trailing digits
    that are NOT a known multi-digit code like RB2/RB3).
    """
    stem = Path(fname).stem  # SMITA00402_R_LB
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[-1].upper()
    return None


def collect_image_records(
    images_dir: str, df_patients: pd.DataFrame, targets: str
) -> list[dict]:
    """
    Walk <images_dir>/<label>/<patient_id>/Unannotated/ for every patient
    and return one dict per image file found, plus one dict per json file.

    Each dict contains:
        patient_id, label, site_code (for images) or 'json_file', abs_path
    """
    records = []

    for _, row in df_patients.iterrows():
        pid = str(row["patient_id"]).strip()
        for label in targets:
            json_filename = None
            unannotated = Path(images_dir) / label / pid / "Unannotated"
            json_path = Path(images_dir) / label / pid / "Json file"
            if not json_path.is_dir():
                json_path = Path(images_dir) / label / pid / "Json"
            if json_path.is_dir():
                json_file = list(json_path.rglob(f"{pid}*"))
                if json_file and len(json_file) != 0:
                    json_filename = json_file[0]
            if unannotated.is_dir():
                img_paths = list(unannotated.rglob(f"{pid}_*"))
                if img_paths and len(img_paths) != 0:
                    for img_path in img_paths:
                        site = _site_code_from_filename(img_path.name)
                        records.append(
                            {
                                "patient_id": pid,
                                "label": label.lower(),
                                "lesion_location": site,
                                "image_path": img_path,
                                "json_file": json_filename,
                            }
                        )

        # label = str(row["label"]).strip()
        # unannotated = Path(images_dir) / label / pid / "Unannotated"
        # if unannotated.exists():

        #     for fpath in sorted(unannotated.iterdir()):
        #         if not fpath.is_file():
        #             continue
        #         if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
        #             continue
        #         site = _site_code_from_filename(fpath.name)
        #         records.append(
        #             {
        #                 "patient_id": pid,
        #                 "label": label,
        #                 "lesion_location": site,  # site code as location
        #                 "image_path": str(fpath),
        #             }
        #         )

        # # JSON annotation file (only one expected per patient)
        # for json_dir_name in ("Json file", "Json"):
        #     json_dir = Path(images_dir) / label / pid / json_dir_name
        #     if json_dir.exists():
        #         for fpath in sorted(json_dir.iterdir()):
        #             if fpath.suffix.lower() == JSON_EXTENSION:
        #                 records.append(
        #                     {
        #                         "patient_id": pid,
        #                         "label": label,
        #                         "lesion_location": "json_file",
        #                         "image_path": str(fpath),
        #                     }
        #                 )
        #         break  # don't double-count if both spellings exist

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Core expander: one metadata row per image
# ─────────────────────────────────────────────────────────────────────────────


def expand_to_image_rows(
    logger: CustomSizeDayRotatingFileHandler,
    df_patients: pd.DataFrame,
    images_dir: str,
    targets: str,
) -> pd.DataFrame:
    """
    For every image / json file found on disk, emit one output row carrying:
      - all patient-level metadata columns
      - label, lesion_location (site code), lesion_present, image_path
      - source, type
    Patient metadata is joined on patient_id.
    """
    image_records = collect_image_records(images_dir, df_patients, targets)
    logger.info(f"  Collected {len(image_records)} image/json records from disk.")

    # Index patient metadata for fast join
    patient_index = df_patients.set_index("patient_id")
    patient_meta_cols = [c for c in df_patients.columns if c != "patient_id"]

    rows = []
    unmatched = set()

    for rec in image_records:
        pid = rec["patient_id"]
        label = rec["label"]

        if pid in patient_index.index:
            meta = patient_index.loc[pid]
            # Handle patients with multiple rows (shouldn't happen in SMART-II
            # but guard anyway — take first)
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            meta_dict = meta.to_dict()
        else:
            unmatched.add(pid)
            meta_dict = {c: np.nan for c in patient_meta_cols}

        # lesion_present: boolean derived from label
        lesion_present = label in ("opmd", "variation")

        row = {
            "patient_id": pid,
            **meta_dict,
            "label": label,
            "lesion_location": rec["lesion_location"],
            "lesion_present": lesion_present,
            "image_path": rec["image_path"],
            "source": "smart_II",
            "type": "original",
            "json_file": rec["json_file"],
        }
        rows.append(row)

    if unmatched:
        logger.info(
            f"  Warning: {len(unmatched)} image patient IDs not in metadata: "
            f"{sorted(unmatched)}"
        )

    df_out = pd.DataFrame(rows)
    logger.info(f"  Expanded to {len(df_out)} rows (one per image/json file).")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Main cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────


def clean_metadata(
    logger: CustomSizeDayRotatingFileHandler,
    xlsx_path: str,
    images_dir: str,
    output_csv: str,
    report_csv: str,
    targets: str,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("  Patient Metadata Cleaning Pipeline  (SMART-II)")
    logger.info("=" * 60)

    # Load
    df_raw = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    logger.info(f"  Raw shape  : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Row 0 is a merged-cell sub-header row — drop it, keep from row 1
    data = df_raw.iloc[1:].copy().reset_index(drop=True)

    # Clean
    data = normalise_nulls(data)
    data = strip_strings(data)
    data = data.rename(columns=COL_RENAME)
    data = normalise_categoricals(data)
    # data["label"] = data.apply(assign_label, axis=1)
    data["patient_id"] = data["patient_id"].astype(str).str.strip()

    # Drop near-empty columns
    before_drop = set(data.columns)
    fully_null_cols = [c for c in data.columns if data[c].isna().all()]
    data = data.drop(columns=fully_null_cols)
    dropped = before_drop - set(data.columns)
    logger.info(f"  Columns dropped ({len(dropped)}): {sorted(dropped)}")

    # Remove rows with no patient_id
    data = data[data["patient_id"].notna() & (data["patient_id"] != "nan")].reset_index(
        drop=True
    )
    logger.info(f"  Patient rows after cleaning: {len(data)}")

    # Expand to one row per image
    logger.info("Expanding to one row per image...")
    df_expanded = expand_to_image_rows(logger, data, images_dir, targets)

    # Column order
    # final_cols = [c for c in COL_ORDER if c in df_expanded.columns]
    # extras = [c for c in df_expanded.columns if c not in final_cols]
    # df_expanded = df_expanded[final_cols + extras]

    # Reports
    logger.info(
        f"  Final shape: {df_expanded.shape[0]} rows × {df_expanded.shape[1]} cols"
    )
    logger.info("  Label distribution:")
    for lbl, cnt in df_expanded["label"].value_counts().items():
        pct = 100 * cnt / len(df_expanded)
        logger.info(f"    {lbl:12s} {cnt:4d}  ({pct:5.1f}%)")

    miss = missingness_report(df_expanded)

    # Save
    os.makedirs(
        os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(report_csv) if os.path.dirname(report_csv) else ".",
        exist_ok=True,
    )
    df_expanded.to_csv(output_csv, index=False)
    miss.to_csv(report_csv, index=False)

    logger.info(f"  Saved CSV    → {output_csv}")
    logger.info(f"  Saved report → {report_csv}")
    logger.info("=" * 60)

    return df_expanded


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def initialize_logger(config):
    log_filename = config.get("LOGGER", "logger.filename")
    return iolog.getLogger(log_filename)


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config=config)

    basepath = config.get("SMART-II-DATAPATH", "smart.basepath")
    patient_metadata = config.get("SMART-II-DATAPATH", "smart.patient.metadata")
    patient_clean_metadata = config.get(
        "SMART-II-DATAPATH", "smart.patient.clean.metadata.filename"
    )
    missing_column_report = config.get(
        "SMART-II-DATAPATH", "smart.patient.missing.filename"
    )
    targets = [
        config.get("SMART-II-DATAPATH", "smart.normal"),
        config.get("SMART-II-DATAPATH", "smart.variation"),
        config.get("SMART-II-DATAPATH", "smart.opmd"),
    ]
    df_clean = clean_metadata(
        logger=logger,
        xlsx_path=f"{basepath}/{patient_metadata}",
        images_dir=basepath,
        output_csv=patient_clean_metadata,
        report_csv=missing_column_report,
        targets=targets,
    )
