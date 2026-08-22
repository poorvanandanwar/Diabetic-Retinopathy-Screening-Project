from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_project_root(start: Path) -> Path:
    """Find the folder containing data/raw."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find data/raw. Open the notebook from the project folder."
    )


def clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def choose_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {clean_name(c): c for c in columns}
    for candidate in candidates:
        if clean_name(candidate) in lookup:
            return lookup[clean_name(candidate)]
    return None


def scan_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_key(dataset: str, image_path: Path, project_root: Path) -> str:
    relative = str(image_path.relative_to(project_root)).replace("\\", "/")
    short_hash = hashlib.sha1(relative.encode()).hexdigest()[:16]
    return f"{dataset}__{short_hash}"


def standard_rows(
    dataset: str,
    images: list[Path],
    root: Path,
    split_from_path,
    labels: dict[str, int] | None = None,
    label_source: str | None = None,
) -> pd.DataFrame:
    labels = labels or {}
    rows = []

    for path in images:
        image_id = path.stem
        label = labels.get(image_id)
        split = split_from_path(path)

        rows.append(
            {
                "dataset": dataset,
                "image_key": image_key(dataset, path, root),
                "image_id": image_id,
                "source_split": split,
                "official_split": split if dataset == "idrid" else pd.NA,
                "image_path": str(path.relative_to(root)).replace("\\", "/"),
                "extension": path.suffix.lower(),
                "label_dr": label,
                "has_label": pd.notna(label),
                "label_source": label_source if label is not None else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def csv_label_map(
    csv_path: Path,
    image_candidates: list[str],
    label_candidates: list[str],
) -> dict[str, int]:
    """Read a label CSV despite harmless column-name differences."""
    if not csv_path.exists():
        return {}

    frame = pd.read_csv(csv_path)
    image_column = choose_column(frame.columns, image_candidates)
    label_column = choose_column(frame.columns, label_candidates)

    print(f"\n{csv_path.name}")
    print(f"Columns found: {list(frame.columns)}")
    print(f"Using image column: {image_column}; label column: {label_column}")

    if image_column is None or label_column is None:
        print("WARNING: Could not identify label columns. Images will remain unlabeled.")
        return {}

    clean = frame[[image_column, label_column]].dropna().copy()
    clean[image_column] = (
        clean[image_column].astype(str)
        .str.replace(r"\.(jpg|jpeg|png|tif|tiff)$", "", regex=True, case=False)
    )
    clean[label_column] = pd.to_numeric(clean[label_column], errors="coerce")
    clean = clean.dropna()

    return dict(
        zip(
            clean[image_column].astype(str),
            clean[label_column].astype(int),
        )
    )


def build_aptos(root: Path) -> pd.DataFrame:
    base = root / "data" / "raw" / "aptos"
    labels = csv_label_map(
        base / "train.csv",
        ["id_code", "image", "image_id"],
        ["diagnosis", "grade", "label", "level"],
    )
    train = standard_rows(
        "aptos",
        scan_images(base / "train_images"),
        root,
        lambda _: "train",
        labels,
        "aptos/train.csv",
    )
    test = standard_rows(
        "aptos",
        scan_images(base / "test_images"),
        root,
        lambda _: "test",
    )
    return pd.concat([train, test], ignore_index=True)


def build_eyepacs(root: Path) -> pd.DataFrame:
    base = root / "data" / "raw" / "eyepacs"
    labels = csv_label_map(
        base / "trainLabels.csv",
        ["image", "image_id", "id_code"],
        ["level", "diagnosis", "grade", "label"],
    )
    return standard_rows(
        "eyepacs",
        scan_images(base / "data"),
        root,
        lambda _: "train",
        labels,
        "eyepacs/trainLabels.csv",
    )


def idrid_split(path: Path) -> str:
    text = str(path).lower()
    if "testing set" in text or "test" in text:
        return "test"
    if "training set" in text or "train" in text:
        return "train"
    return "unknown"


def build_idrid(root: Path) -> pd.DataFrame:
    """
    Use A. Segmentation/Original Images as the canonical IDRiD image location.
    This avoids treating segmentation masks and localization files as retinal images.
    """
    base = root / "data" / "raw" / "idrid"
    original_images = base / "A. Segmentation" / "1. Original Images"
    images = scan_images(original_images)

    grading_files = list((base / "B. Disease Grading").rglob("*.csv"))
    labels: dict[str, int] = {}

    for grading_file in grading_files:
        if "train" in grading_file.name.lower() or "training" in str(grading_file).lower():
            labels.update(
                csv_label_map(
                    grading_file,
                    ["image name", "image", "image_id", "idrid"],
                    ["retinopathy grade", "diagnosis", "grade", "dr grade"],
                )
            )

    # Match image IDs such as IDRiD_01 to CSV IDs such as IDRiD_001.
    def normalise_idrid_id(value):
        number = int(str(value).split("_")[-1])
        return f"IDRiD_{number:03d}"

    canonical_labels = {
        normalise_idrid_id(image_id): grade
        for image_id, grade in labels.items()
    }

    frame = standard_rows(
         "idrid",
        images,
        root,
        idrid_split,
    )

    frame["label_dr"] = frame["image_id"].map(
        lambda image_id: canonical_labels.get(normalise_idrid_id(image_id))
    )

    frame["label_dr"] = pd.array(frame["label_dr"], dtype="Int64")
    frame["has_label"] = frame["label_dr"].notna()
    frame["label_source"] = pd.NA
    frame.loc[frame["has_label"], "label_source"] = (
        "idrid/B. Disease Grading training labels"
    )

    # Never create a random IDRiD split.
    frame["official_split"] = frame["source_split"]
    return frame


def build_messidor2(root: Path) -> pd.DataFrame:
    base = root / "data" / "raw" / "messidor2"
    labels = csv_label_map(
        base / "messidor_data.csv",
        ["image_id", "image", "image name", "id_code", "name"],
        ["retinopathy grade", "diagnosis", "grade", "level", "label"],
    )
    return standard_rows(
        "messidor2",
        scan_images(base / "preprocess"),
        root,
        lambda _: "unknown",
        labels,
        "messidor2/messidor_data.csv",
    )


def build_standard_metadata(root: Path) -> pd.DataFrame:
    frames = [
        build_aptos(root),
        build_eyepacs(root),
        build_idrid(root),
        build_messidor2(root),
    ]
    metadata = pd.concat(frames, ignore_index=True)
    metadata["label_dr"] = pd.array(metadata["label_dr"], dtype="Int64")
    metadata.to_csv(root / "data" / "interim" / "standardized_metadata.csv", index=False)
    return metadata


def validate_one(row: dict, root: Path) -> dict:
    path = root / row["image_path"]
    result = {
        "image_key": row["image_key"],
        "is_readable": False,
        "width": np.nan,
        "height": np.nan,
        "pil_format": pd.NA,
        "validation_error": pd.NA,
    }
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            result["is_readable"] = True
            result["width"], result["height"] = image.size
            result["pil_format"] = image.format
    except (UnidentifiedImageError, OSError, ValueError) as error:
        result["validation_error"] = str(error)[:300]
    return result


def validate_images(metadata: pd.DataFrame, root: Path, workers: int = 1) -> pd.DataFrame:
    records = metadata.to_dict("records")
    if workers == 1:
        results = [validate_one(row, root) for row in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda row: validate_one(row, root), records))

    validation = pd.DataFrame(results)
    validation.to_csv(root / "reports" / "tables" / "image_validation.csv", index=False)
    return validation


def quality_one(row: dict, root: Path) -> dict:
    path = root / row["image_path"]
    result = {
        "image_key": row["image_key"],
        "brightness_mean": np.nan,
        "contrast_std": np.nan,
        "sharpness_laplacian_var": np.nan,
        "dark_pixel_fraction": np.nan,
        "bright_pixel_fraction": np.nan,
        "quality_flag": "unreadable",
    }

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_fraction = float((gray < 15).mean())
    bright_fraction = float((gray > 245).mean())

    flags = []
    if brightness < 25:
        flags.append("very_dark")
    if brightness > 230:
        flags.append("very_bright")
    if contrast < 15:
        flags.append("low_contrast")
    if sharpness < 20:
        flags.append("possibly_blurry")

    result.update(
        {
            "brightness_mean": brightness,
            "contrast_std": contrast,
            "sharpness_laplacian_var": sharpness,
            "dark_pixel_fraction": dark_fraction,
            "bright_pixel_fraction": bright_fraction,
            "quality_flag": ";".join(flags) if flags else "pass",
        }
    )
    return result


def compute_quality_features(metadata: pd.DataFrame, root: Path, workers: int = 1) -> pd.DataFrame:
    records = metadata.to_dict("records")
    if workers == 1:
        results = [quality_one(row, root) for row in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda row: quality_one(row, root), records))

    quality = pd.DataFrame(results)
    quality.to_csv(root / "reports" / "tables" / "image_quality_features.csv", index=False)
    return quality


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = image[:, 1:] > image[:, :-1]
    return f"{int(''.join('1' if x else '0' for x in bits.flatten()), 2):016x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def detect_duplicates(metadata: pd.DataFrame, root: Path) -> pd.DataFrame:
    hashes = []
    for row in metadata.to_dict("records"):
        path = root / row["image_path"]
        try:
            hashes.append(
                {
                    "image_key": row["image_key"],
                    "sha256": sha256_file(path),
                    "dhash": dhash(path),
                }
            )
        except OSError:
            hashes.append({"image_key": row["image_key"], "sha256": pd.NA, "dhash": pd.NA})

    hash_frame = pd.DataFrame(hashes)
    hash_frame.to_csv(root / "reports" / "tables" / "image_hashes.csv", index=False)

    duplicates = []

    # Byte-identical images.
    for _, group in hash_frame.dropna(subset=["sha256"]).groupby("sha256"):
        if len(group) > 1:
            for key in group["image_key"]:
                duplicates.append(
                    {"image_key": key, "duplicate_group": f"exact_{group.name[:12]}", "duplicate_type": "exact"}
                )

    # Near-duplicates based on perceptual dHash, tested only inside hash-prefix buckets.
    valid = hash_frame.dropna(subset=["dhash"]).copy()
    valid["bucket"] = valid["dhash"].str[:4]

    group_number = 0
    for _, group in valid.groupby("bucket"):
        rows = group.to_dict("records")[:250]  # prevents pathological O(n²) buckets
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                if hamming_distance(left["dhash"], right["dhash"]) <= 5:
                    group_number += 1
                    group_id = f"near_{group_number:06d}"
                    duplicates.extend(
                        [
                            {"image_key": left["image_key"], "duplicate_group": group_id, "duplicate_type": "near_dhash"},
                            {"image_key": right["image_key"], "duplicate_group": group_id, "duplicate_type": "near_dhash"},
                        ]
                    )

    duplicate_frame = pd.DataFrame(
        duplicates,
        columns=["image_key", "duplicate_group", "duplicate_type"],
    ).drop_duplicates()

    duplicate_frame.to_csv(root / "reports" / "tables" / "duplicate_candidates.csv", index=False)
    return duplicate_frame


def preprocess_one(row: dict, root: Path, output_root: Path, size: int = 512) -> dict:
    source = root / row["image_path"]
    destination = output_root / row["dataset"] / f"{row['image_key']}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return {"image_key": row["image_key"], "preprocessed_path": pd.NA, "preprocess_status": "failed"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    coordinates = cv2.findNonZero(mask.astype(np.uint8))

    if coordinates is not None:
        x, y, width, height = cv2.boundingRect(coordinates)
        image = image[y:y + height, x:x + width]

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    # Mild CLAHE on luminance: standardized but not destructive.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    image = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "image_key": row["image_key"],
        "preprocessed_path": str(destination.relative_to(root)).replace("\\", "/"),
        "preprocess_status": "ok",
    }


def preprocess_images(metadata: pd.DataFrame, root: Path, size: int = 512) -> pd.DataFrame:
    output_root = root / "data" / "processed" / "images"
    rows = [preprocess_one(row, root, output_root, size) for row in metadata.to_dict("records")]
    result = pd.DataFrame(rows)
    result.to_csv(root / "reports" / "tables" / "preprocessing_manifest.csv", index=False)
    return result


def save_figures(master: pd.DataFrame, root: Path) -> None:
    figures = root / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.countplot(data=master, x="dataset", hue="source_split")
    plt.title("Image count by dataset and source split")
    plt.tight_layout()
    plt.savefig(figures / "dataset_split_counts.png", dpi=160)
    plt.close()

    labeled = master.dropna(subset=["label_dr"]).copy()
    if not labeled.empty:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=labeled, x="label_dr", hue="dataset")
        plt.title("DR grade distribution by dataset")
        plt.tight_layout()
        plt.savefig(figures / "dr_grade_distribution.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=master, x="dataset", y="brightness_mean")
    plt.title("Brightness distribution by dataset")
    plt.tight_layout()
    plt.savefig(figures / "brightness_by_dataset.png", dpi=160)
    plt.close()


def run_phase1(root: Path, workers: int = 1, run_preprocessing: bool = False) -> pd.DataFrame:
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    metadata = build_standard_metadata(root)
    validation = validate_images(metadata, root, workers)
    quality = compute_quality_features(metadata, root, workers)
    duplicates = detect_duplicates(metadata, root)

    master = (
        metadata
        .merge(validation, on="image_key", how="left")
        .merge(quality, on="image_key", how="left")
    )

    if not duplicates.empty:
        summary = (
            duplicates.groupby("image_key")
            .agg(
                duplicate_groups=("duplicate_group", lambda x: ";".join(sorted(set(x)))),
                duplicate_types=("duplicate_type", lambda x: ";".join(sorted(set(x)))),
            )
            .reset_index()
        )
        master = master.merge(summary, on="image_key", how="left")
    else:
        master["duplicate_groups"] = pd.NA
        master["duplicate_types"] = pd.NA

    if run_preprocessing:
        preprocessing = preprocess_images(master, root, size=512)
        master = master.merge(preprocessing, on="image_key", how="left")
    else:
        master["preprocessed_path"] = pd.NA
        master["preprocess_status"] = "not_run"

    master.to_csv(root / "reports" / "tables" / "master_metadata_phase1.csv", index=False)

    summary = (
        master.groupby(["dataset", "source_split"], dropna=False)
        .agg(
            images=("image_key", "count"),
            labeled_images=("has_label", "sum"),
            unreadable_images=("is_readable", lambda x: int((~x.fillna(False)).sum())),
        )
        .reset_index()
    )
    summary.to_csv(root / "reports" / "tables" / "dataset_summary.csv", index=False)

    save_figures(master, root)
    return master