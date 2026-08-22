from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


VALID_GRADES = {0, 1, 2, 3, 4}

GRADE_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def ensure_output_folders(root: Path) -> None:
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "figures").mkdir(parents=True, exist_ok=True)


def create_stratified_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a reproducible train/validation split within each dataset and DR grade.
    IDRiD is handled separately to preserve its official split.
    """
    parts = []

    for (_, _), group in frame.groupby(["dataset", "label_dr_standard"], dropna=False):
        group = group.sample(
            frac=1,
            random_state=random_seed,
        ).copy()

        # Keep very small classes in training rather than creating an empty class.
        validation_count = max(1, round(len(group) * validation_fraction))

        if len(group) <= 1:
            validation_count = 0

        group["model_split"] = "train"
        if validation_count > 0:
            group.iloc[:validation_count, group.columns.get_loc("model_split")] = "validation"

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def standardize_labels_and_splits(
    root: Path,
    validation_fraction: float = 0.20,
    random_seed: int = 42,
) -> pd.DataFrame:
    ensure_output_folders(root)

    input_path = root / "reports" / "tables" / "master_metadata_phase1.csv"
    metadata = pd.read_csv(input_path)

    # Convert labels safely to numeric values.
    metadata["label_dr_standard"] = pd.to_numeric(
        metadata["label_dr"],
        errors="coerce",
    ).astype("Int64")

    metadata["dr_grade_name"] = metadata["label_dr_standard"].map(GRADE_NAMES)

    # An image is eligible only if it is readable and has a valid DR grade.
    metadata["eligible_for_model"] = (
        metadata["is_readable"].fillna(False)
        & metadata["label_dr_standard"].isin(VALID_GRADES)
    )

    # Default: exclude unlabeled APTOS test images and any invalid records.
    metadata["model_split"] = "excluded"
    metadata["split_reason"] = "unlabeled_or_invalid"

    # Preserve IDRiD's official split exactly.
    idrid_train = (
        (metadata["dataset"] == "idrid")
        & (metadata["source_split"] == "train")
        & metadata["eligible_for_model"]
    )
    idrid_test = (
        (metadata["dataset"] == "idrid")
        & (metadata["source_split"] == "test")
        & metadata["eligible_for_model"]
    )

    metadata.loc[idrid_train, "model_split"] = "train"
    metadata.loc[idrid_train, "split_reason"] = "official_idrid_train"

    metadata.loc[idrid_test, "model_split"] = "test"
    metadata.loc[idrid_test, "split_reason"] = "official_idrid_test"

    # APTOS train, EyePACS, and MESSIDOR-2 receive reproducible
    # stratified train/validation splits.
    split_candidates = metadata.loc[
        (metadata["dataset"] != "idrid")
        & metadata["eligible_for_model"]
    ].copy()

    split_candidates = create_stratified_split(
        split_candidates,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    metadata = metadata.drop(
        columns=["model_split", "split_reason"],
    ).merge(
        split_candidates[
            ["image_key", "model_split"]
        ],
        on="image_key",
        how="left",
    )

    metadata["model_split"] = metadata["model_split"].fillna("excluded")

    metadata["split_reason"] = "unlabeled_or_invalid"
    metadata.loc[
        (metadata["dataset"] != "idrid")
        & metadata["eligible_for_model"]
        & (metadata["model_split"] == "train"),
        "split_reason",
    ] = "stratified_train"

    metadata.loc[
        (metadata["dataset"] != "idrid")
        & metadata["eligible_for_model"]
        & (metadata["model_split"] == "validation"),
        "split_reason",
    ] = "stratified_validation"

    metadata.loc[idrid_train, "model_split"] = "train"
    metadata.loc[idrid_train, "split_reason"] = "official_idrid_train"

    metadata.loc[idrid_test, "model_split"] = "test"
    metadata.loc[idrid_test, "split_reason"] = "official_idrid_test"

    # Safety checks.
    assert (
        metadata.loc[
            (metadata["dataset"] == "idrid")
            & (metadata["source_split"] == "test"),
            "model_split",
        ]
        == "test"
    ).all(), "IDRiD test split was changed."

    assert not (
        (metadata["model_split"].isin(["train", "validation", "test"]))
        & (~metadata["eligible_for_model"])
    ).any(), "An ineligible image was included in a model split."

    # Save Phase 2 master metadata.
    output_path = root / "data" / "interim" / "phase2_standardized_metadata.csv"
    metadata.to_csv(output_path, index=False)

    # Label audit table.
    label_audit = (
        metadata.groupby(
            ["dataset", "label_dr_standard", "dr_grade_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="images")
        .sort_values(["dataset", "label_dr_standard"])
    )

    label_audit.to_csv(
        root / "reports" / "tables" / "phase2_label_distribution.csv",
        index=False,
    )

    # Split audit table.
    split_audit = (
        metadata.groupby(
            ["dataset", "model_split", "label_dr_standard"],
            dropna=False,
        )
        .size()
        .reset_index(name="images")
        .sort_values(["dataset", "model_split", "label_dr_standard"])
    )

    split_audit.to_csv(
        root / "reports" / "tables" / "phase2_split_distribution.csv",
        index=False,
    )

    # Grade mapping documentation.
    pd.DataFrame(
        [
            {"label_dr_standard": grade, "dr_grade_name": name}
            for grade, name in GRADE_NAMES.items()
        ]
    ).to_csv(
        root / "reports" / "tables" / "phase2_grade_mapping.csv",
        index=False,
    )

    save_phase2_figure(metadata, root)

    return metadata


def save_phase2_figure(metadata: pd.DataFrame, root: Path) -> None:
    sns.set_theme(style="whitegrid")

    plot_data = metadata.loc[
        metadata["eligible_for_model"]
    ].copy()

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=plot_data,
        x="dr_grade_name",
        hue="dataset",
        order=[
            "No DR",
            "Mild DR",
            "Moderate DR",
            "Severe DR",
            "Proliferative DR",
        ],
    )
    plt.title("Standardized DR Grade Distribution by Dataset")
    plt.xlabel("Standardized DR Grade")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        root / "reports" / "figures" / "phase2_label_distribution.png",
        dpi=160,
    )
    plt.close()