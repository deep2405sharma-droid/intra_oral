from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config
from utils.log_handler import CustomSizeDayRotatingFileHandler

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Column rename map (row-0 + row-1 merged headers → clean snake_case names)
# ---------------------------------------------------------------------------
COL_RENAME = {
    "S.No": "serial_no",
    "SMITA_ID": "patient_id",
    "Age": "age",
    "Sex": "sex",
    "Habit_history": "habit_history",
    "Type_of_habit": "habit_types",
    "Form_of_tobacco": "tobacco_type",
    # Smoking
    "Smoking": "smoking_type",
    "Smoking_Type": "smoking_type",
    "Smoking_Frequency (No of times per day)": "smoking_freq_per_day",
    "Smoking_Duration of Habit (in years)": "smoking_duration_years",
    "Smoking_Habit_status": "smoking_status",
    "Smoking_Start of Habit-Age (in years)": "smoking_onset_age",
    # Chewing
    "Chewing_Frequency (No of times per day)": "chewing_freq_per_day",
    "Chewing_Duration of Habit (in years)": "chewing_duration_years",
    "Chewing_Habit_status": "chewing_habit_status",
    "Chewing_Start of Habit-Age (in years)": "chewing_onset_age",
    # Arecanut
    "Arecanut_Frequency (No of times per day)": "arecanut_freq_per_day",
    "Arecanut_Duration of Habit (in years)": "arecanut_duration_years",
    "Arecanut_Habit_status": "arecanut_status",
    "Arecanut_Start of Habit-Age (in years)": "arecanut_onset_age",
    # Alcohol
    "Alcohol_Frequency (No of times per week)": "alcohol_freq_per_week",
    "Alcohol_Duration of Habit (in years)": "alcohol_duration_years",
    "Alcohol_Habit status": "alcohol_status",
    "Alcohol_Start of Habit-Age (in years)": "alcohol_onset_age",
    # Oral hygiene
    "Brushing_habit": "oral_hygiene_cleaning_aid",
    "Brushing_habit_Type of cleaning aid": "oral_hygiene_cleaning_aid",
    "Brushing_habit_Material used": "oral_hygiene_material",
    "Brushing_habit_Method of brushing": "oral_hygiene_brushing_method",
    "Brushing_habit_Frequency of brushing": "oral_hygiene_brushing_freq",
    "Brushing_habit_Duration of brushing": "oral_hygiene_brushing_duration",
    "Brushing_habit_Frequency of changing toothbrush (in months)": "oral_hygiene_brush_change_freq",
    # Medical / dental history
    "Family history": "family_history",
    "Past Medical history": "past_medical_history",
    "Past Dental History": "past_dental_history",
    "Denture usage": "denture_usage",
}

# Target folder → label value
TARGET_LABEL_MAP = {
    "01. Normal": "normal",
    "02. Variation from normal": "variation",
    "03. OPMD": "opmd",
}

# Region folder name → short column code used throughout
REGION_COL_MAP = {
    "01. Dorsal tongue": "DT",
    "02. Ventral tongue": "VT",
    "03. Left buccal mucosa": "LB",
    "04. Right buccal mucosa": "RB",
    "05. Upper lip": "UL",
    "06. Lower lip": "LL",
    "07. Upper arch": "UA",
    "08. Lower arch": "LA",
    # "09. Json files": "json_file",
}

# Descriptor sheet name → label value it corresponds to
DESCRIPTOR_SHEET_LABEL_MAP = {
    "Variation from normal": "variation",
    "OPMD": "opmd",
}

# Folder that should contribute ONLY json files (no annotated images)
FULL_ANNOTATION_FOLDER = "03. Full annotation"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
JSON_EXTENSION = ".json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_base_patient_id(pid: str) -> str:
    """
    Normalise any variant ID back to its base patient ID for grouping.
    SMITA00220_W  -> SMITA00220
    SMITA00024-1  -> SMITA00024
    SMITA_R_5     -> SMITA_R_5  (no base, keep as-is)
    """
    pid = str(pid).strip()
    # pid = re.sub(r"_W$", "", pid)
    # pid = re.sub(r"-\d+$", "", pid)
    return pid


def normalise_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dash-only and whitespace-only cells with NaN."""
    df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all object/string columns."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def build_merged_header(df_raw: pd.DataFrame) -> list:
    """
    Merge row-0 (group names) and row-1 (sub-column names) into a single
    flat header list, forward-filling the group name across its merged cells.
    """
    row0 = df_raw.iloc[0].tolist()
    row1 = df_raw.iloc[1].tolist()

    filled_group = []
    last = None
    for v in row0:
        if pd.notna(v):
            last = str(v).strip()
        filled_group.append(last)

    headers = []
    for grp, sub in zip(filled_group, row1):
        if pd.isna(sub) or str(sub).strip() == "":
            headers.append(grp)
        else:
            sub_clean = str(sub).strip()
            if grp and not sub_clean.startswith(grp):
                headers.append(f"{grp}_{sub_clean}")
            else:
                headers.append(sub_clean)
    return headers


# ---------------------------------------------------------------------------
# Descriptor loader
# ---------------------------------------------------------------------------


def _parse_descriptor_filename(raw: str) -> tuple | None:
    """
    Parse a descriptor file_name cell into (base_patient_id, region_code).

    Handles three observed patterns:
      SMITA00046_R_DT     -> (SMITA00046, DT)
      SMITA00014_W_LB2    -> (SMITA00014, LB)   [numeric suffix stripped]
      SMITA00023_RT1      -> (SMITA00023, RT)    [no visit-type marker]
      SMITA_R_8 - LB      -> (SMITA_R_8,  LB)   [special R-series patients]
    """
    f = raw.strip().replace(" ", "").replace("__", "_")
    if "SMITA_R_8" in f:
        return "SMITA_R_8", "LB"
    region = f.split("_")[-1]
    region = re.sub(r"\d+$", "", region)

    # Pattern 1: SMITA<id>_<R|W>_<REGION><digit?>
    # m = re.match(r"^SMITA[^_-]*?(?:[_-][W1])_([A-Z]+)\d*$", f, re.IGNORECASE)
    # if m:
    #     pid = get_base_patient_id(m.group(1))
    #     return pid, m.group(2).upper()
    # # Pattern 2: SMITA<id>_<REGION><digit?>  (no visit-type marker)
    # m2 = re.match(r"(SMITA\w+?)_([A-Z]+)\d*$", f, re.IGNORECASE)
    # if m2:
    #     pid = get_base_patient_id(m2.group(1))
    #     return pid, m2.group(2).upper()
    # # Pattern 3: SMITA_R_<N>_<REGION>  (R-series, already normalised above)
    # m3 = re.match(r"(SMITA_R_\d+)_([A-Z]+)", f, re.IGNORECASE)
    # if m3:
    #     return m3.group(1), m3.group(2).upper()
    pid_match = re.match(r"^SMITA[^_-]*?(?:[_-][W1])", f, re.IGNORECASE)
    if not pid_match:
        pid_match = re.match(r"([^_.]+)", f)
    if pid_match:
        return pid_match.group(0), region
    return None


def load_lesion_classification_lookup(descriptor_path: str) -> dict:
    """
    Build a lookup dict:
        (base_patient_id, region_code, label) -> lesion_classification

    Reads the 'Variation from normal' and 'OPMD' sheets only.
    Column layout: col-0=S.No, col-1=File_name, col-2=classification
    """
    lookup: dict = {}
    for sheet, label in DESCRIPTOR_SHEET_LABEL_MAP.items():
        df = pd.read_excel(descriptor_path, sheet_name=sheet, header=None)
        for _, row in df.iloc[1:].iterrows():  # skip header row
            raw_fname = str(row[1]).strip() if pd.notna(row[1]) else ""
            classification = str(row[2]).strip() if pd.notna(row[2]) else ""
            if not raw_fname or raw_fname == "nan":
                continue
            parsed = _parse_descriptor_filename(raw_fname)
            print(f"parsed: {parsed}")
            if parsed:
                pid, region = parsed
                lookup[(pid, region, label)] = classification
    return lookup


# ---------------------------------------------------------------------------
# Image collector  →  one record per image file
# ---------------------------------------------------------------------------


def collect_image_records(
    basepath: str, targets: list, folders: list, regions: list, json_folder: str
) -> list[dict]:
    """
    Walk the directory tree and return a flat list of dicts, one per image file.

    Rules applied here:
      - From FULL_ANNOTATION_FOLDER  : collect ONLY .json files (no images).
      - From all other folders        : collect image files only (no .json).

    Each record contains:
        patient_id, target, folder, region, region_col, abs_path, filename
    """
    records = []

    for target in targets:
        for folder in folders:
            for region in regions:
                json_file = None
                region_col = REGION_COL_MAP.get(region)
                dir_path = os.path.join(basepath, target, folder, region)
                if not os.path.isdir(dir_path):
                    continue

                # is_full_annotation = folder == FULL_ANNOTATION_FOLDER

                for fname in sorted(os.listdir(dir_path)):
                    fpath = os.path.join(dir_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    ext = Path(fname).suffix.lower()

                    # Folder-specific filtering
                    # if is_full_annotation:
                    #     if ext != JSON_EXTENSION:
                    #         continue  # skip annotated images
                    # else:
                    if ext not in IMAGE_EXTENSIONS:
                        continue  # skip non-images
                    if "SMITA_R_8" in fname:
                        base_pid = "SMITA_R_8"
                    else:
                        pid_match = re.match(
                            r"^SMITA[^_-]*?(?:[_-][W1])", fname, re.IGNORECASE
                        )
                        if not pid_match:
                            pid_match = re.match(r"([^_.]+)", fname)
                            if not pid_match:
                                continue
                        base_pid = pid_match.group(0).strip()  # get_base_patient_id(
                    json_filepath = Path(f"{basepath}/{target}/{json_folder}/full json")
                    if not json_filepath.is_dir():
                        json_filepath = Path(
                            f"{basepath}/{target}/{json_folder}/09. Json files"
                        )
                    if json_filepath.is_dir():
                        json_filename = list(json_filepath.rglob(f"{base_pid}_*"))
                        if json_filename and len(json_filename) != 0:
                            json_file = json_filename[0]
                    print(f"{base_pid}: {json_file}")
                    records.append(
                        {
                            "patient_id": base_pid,
                            "target": target,
                            "folder": folder,
                            "region": region,
                            "region_col": region_col,
                            "abs_path": fpath,
                            "filename": fname,
                            "json_file": json_file,
                        }
                    )

    return records


# ---------------------------------------------------------------------------
# Core expander: one metadata row per image
# ---------------------------------------------------------------------------


def expand_metadata_to_image_rows(
    logger: CustomSizeDayRotatingFileHandler,
    df_patients: pd.DataFrame,
    basepath: str,
    targets: list,
    folders: list,
    regions: list,
    json_folder: str,
    descriptor_lookup: dict,
) -> pd.DataFrame:
    """
    For every image record on disk, emit one output row that carries:
      - all patient-level metadata columns (joined on patient_id)
      - label, lesion_location, lesion_present, lesion_classification
      - image_path  (absolute path to the image / json file)
      - source, type
    """
    image_records = collect_image_records(
        basepath, targets, folders, regions, json_folder
    )
    logger.info(f"  Collected {len(image_records)} image/json records from disk.")

    # Index patient metadata by base patient_id for fast lookup
    df_patients = df_patients.copy()
    # df_patients["patient_id"] = df_patients["patient_id"].apply(
    #     lambda x: get_base_patient_id(str(x)) if pd.notna(x) else x
    # )
    patient_index = df_patients.set_index("patient_id")

    # Patient-level columns (everything except the old region-bucket columns if any)
    patient_cols = [c for c in df_patients.columns if c != "patient_id"]

    rows = []
    unmatched_pids = set()

    for rec in image_records:
        pid = rec["patient_id"]
        target = rec["target"]
        region_col = rec["region_col"]
        json_file = rec["json_file"]

        # Derive label from the target folder
        label = TARGET_LABEL_MAP.get(target, np.nan)

        # Pull patient-level metadata
        if pid in patient_index.index:
            patient_row = patient_index.loc[pid]
            # If there are multiple rows for the same patient (watchlist / repeat visit),
            # take the first one; longitudinal analysis can use the raw xlsx directly.
            if isinstance(patient_row, pd.DataFrame):
                patient_row = patient_row.iloc[0]
            meta = patient_row.to_dict()
        else:
            unmatched_pids.add(pid)
            meta = {c: np.nan for c in patient_cols}

        # Lesion classification from descriptor (only for variation / opmd)
        lesion_classification = np.nan
        if label in ("variation", "opmd") and region_col:
            lesion_classification = descriptor_lookup.get(
                (pid, region_col, label), np.nan
            )

        row = {
            "patient_id": pid,
            **meta,
            "label": label,
            "lesion_location": region_col if region_col else np.nan,
            "lesion_present": label in ("variation", "opmd"),
            "lesion_classification": lesion_classification,
            "image_path": rec["abs_path"],
            "json_file": json_file,
            "source": "smart_om",
            "type": "original",
        }
        rows.append(row)

    if unmatched_pids:
        logger.info(
            f"  Warning: {len(unmatched_pids)} image patient IDs not found in "
            f"metadata xlsx: {sorted(unmatched_pids)}"
        )

    df_out = pd.DataFrame(rows)
    logger.info(f"  Expanded to {len(df_out)} rows (one per image/json file).")
    return df_out


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def getParams(config) -> tuple:
    basepath = config.get("SMART-OM-DATAPATH", "smartom.basepath")
    output_csv = config.get(
        "SMART-OM-DATAPATH", "smartom.patient.clean.metadata.filename"
    )
    report_csv = config.get("SMART-OM-DATAPATH", "smartom.patient.missing.filename")

    folders = [
        f.strip() for f in config.get("SMART-OM-DATAPATH", "smartom.folders").split(",")
    ]
    regions = [
        r.strip() for r in config.get("SMART-OM-DATAPATH", "smartom.regions").split(",")
    ]
    json_folder = config.get("SMART-OM-DATAPATH", "smartom.json.folder")
    targets = [
        config.get("SMART-OM-DATAPATH", "smartom.normal"),
        config.get("SMART-OM-DATAPATH", "smartom.variation"),
        config.get("SMART-OM-DATAPATH", "smartom.opmd"),
    ]
    descriptor_path = (
        f"{basepath}/{config.get('SMART-OM-DATAPATH', 'smartom.descriptor.filename')}"
    )
    return (
        basepath,
        targets,
        folders,
        regions,
        output_csv,
        report_csv,
        descriptor_path,
        json_folder,
    )


def clean_metadata(
    logger: CustomSizeDayRotatingFileHandler,
    xlsx_path: str,
    config,
) -> pd.DataFrame:
    logger.info(f"Reading: {xlsx_path}")
    df_raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    logger.info(f"Raw shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")

    (
        basepath,
        targets,
        folders,
        regions,
        output_csv,
        report_csv,
        descriptor_path,
        json_folder,
    ) = getParams(config)

    # Build flat header from merged rows 0 + 1
    merged_headers = build_merged_header(df_raw)

    # Data starts at row 2
    data = df_raw.iloc[2:].copy().reset_index(drop=True)
    data.columns = merged_headers

    # Null normalisation & string stripping
    data = normalise_nulls(data)
    data = strip_strings(data)

    # Rename columns to snake_case
    rename_map = {h: COL_RENAME[h] for h in data.columns if h in COL_RENAME}
    print(f"Rename map: {rename_map}")
    data.rename(columns=rename_map, inplace=True)
    logger.info(f"Renamed {len(rename_map)} columns.")

    # Drop ONLY fully-null columns
    fully_null_cols = [c for c in data.columns if data[c].isna().all()]
    data.drop(columns=fully_null_cols, inplace=True)
    logger.info(f"Dropped {len(fully_null_cols)} fully-null columns: {fully_null_cols}")

    # Save column missingness report
    os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)
    miss_report = pd.DataFrame(
        {
            "column": data.columns,
            "null_count": data.isna().sum().values,
            "null_pct": (data.isna().sum() / len(data) * 100).round(2).values,
        }
    )
    miss_report.to_csv(report_csv, index=False)
    logger.info(f"Missingness report saved -> {report_csv}")

    # Load lesion classification lookup from descriptor file
    logger.info(f"Loading descriptor: {descriptor_path}")
    descriptor_lookup = load_lesion_classification_lookup(descriptor_path)
    logger.info(f"  Descriptor lookup built: {len(descriptor_lookup)} entries.")

    # Expand: one row per image, with all metadata + new columns attached
    logger.info("Expanding to one row per image...")
    print(data.columns)
    df_expanded = expand_metadata_to_image_rows(
        logger=logger,
        df_patients=data,
        basepath=basepath,
        targets=targets,
        folders=folders,
        regions=regions,
        json_folder=json_folder,
        descriptor_lookup=descriptor_lookup,
    )

    # Save clean CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_expanded.to_csv(output_csv, index=False)
    logger.info(f"Clean metadata saved -> {output_csv}")
    logger.info(
        f"Final shape: {df_expanded.shape[0]} rows x {df_expanded.shape[1]} cols"
    )

    return df_expanded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def initialize_logger(config):
    log_filename = config.get("LOGGER", "logger.filename")
    logger = iolog.getLogger(log_filename)
    return logger


if __name__ == "__main__":
    config = load_config()
    logger = initialize_logger(config=config)

    basepath = config.get("SMART-OM-DATAPATH", "smartom.basepath")
    metadata_file = config.get("SMART-OM-DATAPATH", "smartom.metadata.file")
    xlsx_path = os.path.join(basepath, metadata_file)

    df_clean = clean_metadata(
        logger=logger,
        xlsx_path=xlsx_path,
        config=config,
    )
