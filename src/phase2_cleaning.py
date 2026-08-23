from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_folders(root: Path) -> None:
    for folder in [
        root / "data" / "interim",
        root / "data" / "processed" / "images",
        root / "reports" / "tables",
        root / "reports" / "figures",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def run_parallel(function, records, workers: int, label: str):
    """Run image work in parallel and print progress every 1,000 images."""
    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, record) for record in records]

        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())

            if completed % 1000 == 0 or completed == len(records):
                print(f"{label}: {completed:,}/{len(records):,}")

    return results


def save_corrupted_images(root: Path) -> pd.DataFrame:
    """Create corrupted_images.csv from Phase 1 validation results."""
    validation_path = root / "reports" / "tables" / "image_validation.csv"
    master_path = root / "reports" / "tables" / "master_metadata_phase1.csv"

    validation = pd.read_csv(validation_path)
    master = pd.read_csv(master_path)

    merged = master.merge(
        validation[
            [
                "image_key",
                "is_readable",
                "validation_error",
            ]
        ],
        on="image_key",
        how="left",
        suffixes=("", "_validation"),
    )

    readable_column = (
        "is_readable_validation"
        if "is_readable_validation" in merged.columns
        else "is_readable"
    )

    error_column = (
        "validation_error_validation"
        if "validation_error_validation" in merged.columns
        else "validation_error"
    )

    corrupted = merged.loc[
        ~merged[readable_column].fillna(False),
        [
            "image_key",
            "image_id",
            "dataset",
            "image_path",
            error_column,
        ],
    ].copy()

    corrupted = corrupted.rename(
        columns={error_column: "reason"}
    )

    corrupted.to_csv(
        root / "reports" / "tables" / "corrupted_images.csv",
        index=False,
    )

    print(f"Corrupted/unreadable images recorded: {len(corrupted)}")
    return corrupted


def quality_one(root: Path, row: dict) -> dict:
    """Calculate automatic quality features at reduced resolution for speed."""
    path = root / row["image_path"]

    result = {
        "image_key": row["image_key"],
        "brightness_mean": np.nan,
        "contrast_std": np.nan,
        "sharpness_laplacian_var": np.nan,
        "dark_pixel_fraction": np.nan,
        "bright_pixel_fraction": np.nan,
        "quality_flag_auto": "unreadable",
    }

    # Reduced-resolution loading is sufficient for these quality measures.
    gray = cv2.imread(
        str(path),
        cv2.IMREAD_REDUCED_GRAYSCALE_4,
    )

    if gray is None:
        return result

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
            "quality_flag_auto": ";".join(flags) if flags else "pass",
        }
    )

    return result


def add_official_quality_labels(root: Path, quality: pd.DataFrame) -> pd.DataFrame:
    """
    Add the official MESSIDOR-2 gradability label where it exists.
    Other datasets remain missing rather than being guessed as 'good'.
    """
    master = pd.read_csv(
        root / "reports" / "tables" / "master_metadata_phase1.csv"
    )

    messidor_csv = root / "data" / "raw" / "messidor2" / "messidor_data.csv"
    messidor = pd.read_csv(messidor_csv)

    source_quality = {}
    if (
        "id_code" in messidor.columns
        and "adjudicated_gradable" in messidor.columns
    ):
        source_quality = dict(
            zip(
                messidor["id_code"].astype(str),
                pd.to_numeric(
                    messidor["adjudicated_gradable"],
                    errors="coerce",
                ),
            )
        )

    master["quality_label"] = pd.NA
    master["quality_label_source"] = pd.NA

    messidor_mask = master["dataset"] == "messidor2"

    master.loc[messidor_mask, "quality_label"] = (
        master.loc[messidor_mask, "image_id"]
        .map(source_quality)
    )

    master.loc[
        messidor_mask & master["quality_label"].notna(),
        "quality_label_source",
    ] = "messidor2/adjudicated_gradable"

    quality = quality.merge(
        master[
            [
                "image_key",
                "quality_label",
                "quality_label_source",
            ]
        ],
        on="image_key",
        how="left",
    )

    quality["quality_label"] = pd.array(
        quality["quality_label"],
        dtype="Int64",
    )

    return quality


def run_quality_analysis(root: Path, workers: int = 4) -> pd.DataFrame:
    make_folders(root)

    master = pd.read_csv(
        root / "reports" / "tables" / "master_metadata_phase1.csv"
    )

    records = master.loc[
        master["is_readable"].fillna(False)
    ].to_dict("records")

    def work(row):
        return quality_one(root, row)

    results = run_parallel(
        work,
        records,
        workers=workers,
        label="Quality analysis",
    )

    quality = pd.DataFrame(results)
    quality = add_official_quality_labels(root, quality)

    quality.to_csv(
        root / "reports" / "tables" / "image_quality_features.csv",
        index=False,
    )

    summary = (
        quality.groupby("quality_flag_auto", dropna=False)
        .size()
        .reset_index(name="images")
        .sort_values("images", ascending=False)
    )

    summary.to_csv(
        root / "reports" / "tables" / "quality_summary_by_flag.csv",
        index=False,
    )

    return quality


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def dhash(path: Path) -> str | None:
    """Create a 64-bit perceptual difference hash."""
    image = cv2.imread(
        str(path),
        cv2.IMREAD_REDUCED_GRAYSCALE_4,
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )

    bits = image[:, 1:] > image[:, :-1]

    return f"{int(''.join('1' if bit else '0' for bit in bits.flatten()), 2):016x}"


def hash_one(root: Path, row: dict) -> dict:
    path = root / row["image_path"]

    try:
        return {
            "image_key": row["image_key"],
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "sha256": sha256_file(path),
            "perceptual_hash": dhash(path),
        }
    except OSError:
        return {
            "image_key": row["image_key"],
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "sha256": pd.NA,
            "perceptual_hash": pd.NA,
        }


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root != second_root:
            self.parent[second_root] = first_root


def detect_duplicates(root: Path, workers: int = 4) -> pd.DataFrame:
    """
    Detect exact duplicates with SHA256 and near-duplicates with perceptual hash.

    The near-duplicate search avoids comparing every image with every other image.
    """
    make_folders(root)

    master = pd.read_csv(
        root / "reports" / "tables" / "master_metadata_phase1.csv"
    )

    records = master.loc[
        master["is_readable"].fillna(False)
    ].to_dict("records")

    def work(row):
        return hash_one(root, row)

    hashes = pd.DataFrame(
        run_parallel(
            work,
            records,
            workers=workers,
            label="Image hashing",
        )
    )

    hashes.to_csv(
        root / "reports" / "tables" / "image_hashes.csv",
        index=False,
    )

    keys = hashes["image_key"].tolist()
    groups = UnionFind(keys)
    image_types = {key: set() for key in keys}

    # Exact duplicate groups.
    for _, group in hashes.dropna(subset=["sha256"]).groupby("sha256"):
        group_keys = group["image_key"].tolist()

        if len(group_keys) > 1:
            first = group_keys[0]

            for key in group_keys[1:]:
                groups.union(first, key)

            for key in group_keys:
                image_types[key].add("exact_sha256")

    # Efficient near-duplicate candidate search.
    # A Hamming distance <= 5 must have at least one 16-bit quarter
    # differing by at most one bit.
    index = {}

    valid_hashes = hashes.dropna(
        subset=["perceptual_hash"]
    ).to_dict("records")

    for current in valid_hashes:
        value = int(current["perceptual_hash"], 16)

        for quarter in range(4):
            shift = (3 - quarter) * 16
            block = (value >> shift) & 0xFFFF

            neighbouring_blocks = [
                block,
                *[block ^ (1 << bit) for bit in range(16)],
            ]

            for candidate_block in neighbouring_blocks:
                for earlier in index.get((quarter, candidate_block), []):
                    if (
                        hamming_distance(
                            current["perceptual_hash"],
                            earlier["perceptual_hash"],
                        )
                        <= 5
                    ):
                        groups.union(
                            current["image_key"],
                            earlier["image_key"],
                        )

                        image_types[current["image_key"]].add(
                            "near_perceptual_hash"
                        )
                        image_types[earlier["image_key"]].add(
                            "near_perceptual_hash"
                        )

            index.setdefault((quarter, block), []).append(current)

    components = {}

    for key in keys:
        root_key = groups.find(key)
        components.setdefault(root_key, []).append(key)

    duplicate_rows = []
    group_number = 0

    for component in components.values():
        if len(component) <= 1:
            continue

        group_number += 1
        duplicate_group = f"duplicate_{group_number:05d}"

        for key in component:
            row = hashes.loc[
                hashes["image_key"] == key
            ].iloc[0]

            duplicate_rows.append(
                {
                    "duplicate_group": duplicate_group,
                    "image_key": key,
                    "image_id": row["image_id"],
                    "dataset": row["dataset"],
                    "duplicate_type": ";".join(
                        sorted(image_types[key])
                    ),
                    "review_status": "pending_review",
                }
            )

    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=[
            "duplicate_group",
            "image_key",
            "image_id",
            "dataset",
            "duplicate_type",
            "review_status",
        ],
    )

    duplicates.to_csv(
        root / "reports" / "tables" / "duplicate_candidates.csv",
        index=False,
    )

    print(
        f"Images in duplicate groups: {len(duplicates):,}"
    )

    return duplicates


def preprocess_one(
    root: Path,
    output_root: Path,
    row: dict,
    size: int,
) -> dict:
    source = root / row["image_path"]
    destination = (
        output_root
        / row["dataset"]
        / f"{row['image_key']}.jpg"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return {
            "image_key": row["image_key"],
            "processed_path": str(
                destination.relative_to(root)
            ).replace("\\", "/"),
            "processed_width": size,
            "processed_height": size,
            "processed_format": "JPEG",
            "preprocess_status": "already_exists",
        }

    image = cv2.imread(
        str(source),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return {
            "image_key": row["image_key"],
            "processed_path": pd.NA,
            "processed_width": pd.NA,
            "processed_height": pd.NA,
            "processed_format": pd.NA,
            "preprocess_status": "failed",
        }

    # Remove black border/background.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    points = cv2.findNonZero(mask.astype(np.uint8))

    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        image = image[y:y + height, x:x + width]

    # JPEG output is three-channel colour and model-ready.
    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    saved = cv2.imwrite(
        str(destination),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    return {
        "image_key": row["image_key"],
        "processed_path": (
            str(destination.relative_to(root)).replace("\\", "/")
            if saved
            else pd.NA
        ),
        "processed_width": size if saved else pd.NA,
        "processed_height": size if saved else pd.NA,
        "processed_format": "JPEG" if saved else pd.NA,
        "preprocess_status": "ok" if saved else "failed",
    }


def preprocess_images(
    root: Path,
    size: int = 224,
    workers: int = 4,
) -> pd.DataFrame:
    make_folders(root)

    master = pd.read_csv(
        root / "reports" / "tables" / "master_metadata_phase1.csv"
    )

    output_root = root / "data" / "processed" / "images"

    records = master.loc[
        master["is_readable"].fillna(False)
    ].to_dict("records")

    def work(row):
        return preprocess_one(
            root,
            output_root,
            row,
            size,
        )

    manifest = pd.DataFrame(
        run_parallel(
            work,
            records,
            workers=workers,
            label="Preprocessing images",
        )
    )

    manifest.to_csv(
        root / "reports" / "tables" / "preprocessing_manifest.csv",
        index=False,
    )

    return manifest


def make_duplicate_safe_split(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Keep every duplicate group in one split.

    If a duplicate of an official test image appears elsewhere, the non-test
    image is excluded to prevent leakage into training/validation.
    """
    metadata = metadata.copy()

    if "duplicate_group" not in metadata.columns:
        return metadata

    for group_name, group in metadata.dropna(
        subset=["duplicate_group"]
    ).groupby("duplicate_group"):

        splits = set(group["model_split"].dropna())

        if "test" in splits:
            non_test_indexes = group.index[
                group["model_split"] != "test"
            ]

            metadata.loc[
                non_test_indexes,
                "model_split",
            ] = "excluded"

            metadata.loc[
                non_test_indexes,
                "split_reason",
            ] = "duplicate_of_test_image"

        elif "train" in splits:
            metadata.loc[
                group.index,
                "model_split",
            ] = "train"

            metadata.loc[
                group.index,
                "split_reason",
            ] = "duplicate_group_train"

        elif "validation" in splits:
            metadata.loc[
                group.index,
                "model_split",
            ] = "validation"

            metadata.loc[
                group.index,
                "split_reason",
            ] = "duplicate_group_validation"
            
    # Safety rule: never use unlabeled or invalid images for modelling,
    # even if they belong to a duplicate group.
    metadata.loc[
        ~metadata["eligible_for_model"].fillna(False),
        "model_split",
    ] = "excluded"

    metadata.loc[
        ~metadata["eligible_for_model"].fillna(False),
        "split_reason",
    ] = "unlabeled_or_invalid"

    return metadata


def build_annotation_availability(root: Path) -> pd.DataFrame:
    """
    Starting inventory. Confirm any 'Unknown' values against each dataset's
    official documentation before using those annotations in research.
    """
    table = pd.DataFrame(
        [
            {
                "dataset": "aptos",
                "dr_grade": "Available",
                "quality_label": "Unknown",
                "dme_label": "Not available",
                "lesion_boxes": "Not available",
                "lesion_masks": "Not available",
            },
            {
                "dataset": "eyepacs",
                "dr_grade": "Available",
                "quality_label": "Unknown",
                "dme_label": "Not available",
                "lesion_boxes": "Not available",
                "lesion_masks": "Not available",
            },
            {
                "dataset": "idrid",
                "dr_grade": "Available",
                "quality_label": "Unknown",
                "dme_label": "Available in disease-grading labels",
                "lesion_boxes": "Verify separately",
                "lesion_masks": "Available in segmentation data",
            },
            {
                "dataset": "messidor2",
                "dr_grade": "Available",
                "quality_label": "Available: adjudicated_gradable",
                "dme_label": "Available in source CSV",
                "lesion_boxes": "Not available",
                "lesion_masks": "Not available",
            },
        ]
    )

    table.to_csv(
        root / "reports" / "tables" / "annotation_availability.csv",
        index=False,
    )

    return table


def build_cleaned_master_metadata(root: Path) -> pd.DataFrame:
    """
    Merge Phase 2 label/split metadata with cleaning outputs.
    Run this only after quality analysis, duplicate detection,
    and preprocessing have finished.
    """
    phase2 = pd.read_csv(
        root / "data" / "interim" / "phase2_standardized_metadata.csv"
    )

    quality = pd.read_csv(
        root / "reports" / "tables" / "image_quality_features.csv"
    )

    duplicates = pd.read_csv(
        root / "reports" / "tables" / "duplicate_candidates.csv"
    )

    preprocessing = pd.read_csv(
        root / "reports" / "tables" / "preprocessing_manifest.csv"
    )

    final = phase2.merge(
        quality,
        on="image_key",
        how="left",
    ).merge(
        duplicates[
            [
                "image_key",
                "duplicate_group",
                "duplicate_type",
                "review_status",
            ]
        ],
        on="image_key",
        how="left",
    ).merge(
        preprocessing,
        on="image_key",
        how="left",
    )

    final = make_duplicate_safe_split(final)

    final.to_csv(
        root / "reports" / "tables" / "final_cleaned_metadata.csv",
        index=False,
    )

    split_summary = (
        final.groupby(
            ["dataset", "model_split", "label_dr_standard"],
            dropna=False,
        )
        .size()
        .reset_index(name="images")
    )

    split_summary.to_csv(
        root / "reports" / "tables" / "final_split_distribution.csv",
        index=False,
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=final.loc[
            final["model_split"].isin(
                ["train", "validation", "test"]
            )
        ],
        x="model_split",
        hue="dataset",
    )
    plt.title("Final Duplicate-Aware Dataset Split")
    plt.tight_layout()
    plt.savefig(
        root / "reports" / "figures" / "final_split_distribution.png",
        dpi=160,
    )
    plt.close()

    return final

def detect_duplicates(root: Path, workers: int = 4) -> pd.DataFrame:
    """
    Safe duplicate policy:

    - SHA256 exact matches are automatically treated as duplicates.
    - Perceptual hashes are saved only as review information.
    - Near perceptual matches never automatically exclude images or change splits.
    """
    make_folders(root)

    master = pd.read_csv(
        root / "reports" / "tables" / "master_metadata_phase1.csv"
    )

    hash_path = root / "reports" / "tables" / "image_hashes.csv"

    expected_keys = set(master["image_key"])

    # Reuse the completed 42,541-image hashing work if possible.
    if hash_path.exists():
        hashes = pd.read_csv(hash_path)

        if set(hashes["image_key"]) == expected_keys:
            print("Reusing existing image_hashes.csv; no images will be hashed again.")
        else:
            print("Hash file does not match metadata; hashing images again.")

            records = master.loc[
                master["is_readable"].fillna(False)
            ].to_dict("records")

            def work(row):
                return hash_one(root, row)

            hashes = pd.DataFrame(
                run_parallel(
                    work,
                    records,
                    workers=workers,
                    label="Image hashing",
                )
            )

            hashes.to_csv(hash_path, index=False)

    else:
        records = master.loc[
            master["is_readable"].fillna(False)
        ].to_dict("records")

        def work(row):
            return hash_one(root, row)

        hashes = pd.DataFrame(
            run_parallel(
                work,
                records,
                workers=workers,
                label="Image hashing",
            )
        )

        hashes.to_csv(hash_path, index=False)

    # Only byte-for-byte identical files are automatic duplicates.
    duplicate_rows = []

    for sha256, group in hashes.dropna(subset=["sha256"]).groupby("sha256"):
        if len(group) <= 1:
            continue

        duplicate_group = f"exact_{sha256[:12]}"

        for _, row in group.iterrows():
            duplicate_rows.append(
                {
                    "duplicate_group": duplicate_group,
                    "image_key": row["image_key"],
                    "image_id": row["image_id"],
                    "dataset": row["dataset"],
                    "duplicate_type": "exact_sha256",
                    "review_status": "automatic_exact_match",
                }
            )

    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=[
            "duplicate_group",
            "image_key",
            "image_id",
            "dataset",
            "duplicate_type",
            "review_status",
        ],
    )

    duplicates.to_csv(
        root / "reports" / "tables" / "duplicate_candidates.csv",
        index=False,
    )

    # Perceptual hashes are retained for later manual review only.
    perceptual_review = (
        hashes.dropna(subset=["perceptual_hash"])
        .groupby("perceptual_hash")
        .agg(
            images=("image_key", "count"),
            datasets=("dataset", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )

    perceptual_review = perceptual_review.loc[
        perceptual_review["images"] > 1
    ].sort_values("images", ascending=False)

    perceptual_review.to_csv(
        root / "reports" / "tables" / "perceptual_hash_review_summary.csv",
        index=False,
    )

    print(
        f"Exact-duplicate images: {len(duplicates):,}"
    )
    print(
        f"Exact-duplicate groups: "
        f"{duplicates['duplicate_group'].nunique() if not duplicates.empty else 0:,}"
    )
    print(
        "Perceptual-hash matches were saved for manual review only."
    )

    return duplicates