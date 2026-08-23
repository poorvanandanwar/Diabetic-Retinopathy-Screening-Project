from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


GRADE_ORDER = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


def make_output_folders(root: Path) -> tuple[Path, Path]:
    figures_dir = root / "reports" / "figures" / "phase5_eda"
    tables_dir = root / "reports" / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return figures_dir, tables_dir


def load_metadata(root: Path) -> pd.DataFrame:
    metadata_path = root / "data" / "metadata" / "unified_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "unified_metadata.csv was not found. "
            "Run Phase 3/4 metadata enrichment first."
        )

    metadata = pd.read_csv(metadata_path)

    metadata["label_dr_standard"] = pd.to_numeric(
        metadata["label_dr_standard"],
        errors="coerce",
    )

    metadata["quality_label"] = pd.to_numeric(
        metadata["quality_label"],
        errors="coerce",
    )

    return metadata


def save_dataset_summary(metadata: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    summary = (
        metadata.groupby("dataset", dropna=False)
        .agg(
            images=("image_key", "count"),
            labeled_images=("label_dr_standard", lambda values: values.notna().sum()),
            usable_model_images=(
                "model_split",
                lambda values: values.isin(
                    ["train", "validation", "test"]
                ).sum(),
            ),
            official_quality_labels=(
                "quality_label",
                lambda values: values.notna().sum(),
            ),
            dme_labels=(
                "dme_label",
                lambda values: values.notna().sum(),
            ),
            lesion_masks=(
                "lesion_available",
                lambda values: values.fillna(False).sum(),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        tables_dir / "phase5_eda_dataset_summary.csv",
        index=False,
    )

    return summary


def plot_dr_class_distribution(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plot_data = metadata.dropna(
        subset=["dr_grade_name"]
    ).copy()

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=plot_data,
        x="dr_grade_name",
        order=GRADE_ORDER,
        color="#3B82F6",
    )
    plt.title("DR Grade Distribution Across All Datasets")
    plt.xlabel("DR Grade")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "01_dr_class_distribution.png",
        dpi=160,
    )
    plt.close()


def plot_dr_grade_by_dataset(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plot_data = metadata.dropna(
        subset=["dr_grade_name"]
    ).copy()

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=plot_data,
        x="dr_grade_name",
        hue="dataset",
        order=GRADE_ORDER,
    )
    plt.title("DR Grade Distribution by Dataset")
    plt.xlabel("DR Grade")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "02_dr_grade_by_dataset.png",
        dpi=160,
    )
    plt.close()


def plot_images_per_dataset(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    counts = (
        metadata.groupby("dataset")
        .size()
        .reset_index(name="images")
        .sort_values("images", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=counts,
        x="dataset",
        y="images",
        color="#10B981",
    )
    plt.title("Number of Images per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Images")
    plt.tight_layout()

    plt.savefig(
        figures_dir / "03_images_per_dataset.png",
        dpi=160,
    )
    plt.close()


def plot_resolution_distribution(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    resolution_data = metadata.dropna(
        subset=["width", "height"]
    ).copy()

    resolution_data["width"] = pd.to_numeric(
        resolution_data["width"],
        errors="coerce",
    )

    resolution_data["height"] = pd.to_numeric(
        resolution_data["height"],
        errors="coerce",
    )

    resolution_data["megapixels"] = (
        resolution_data["width"]
        * resolution_data["height"]
        / 1_000_000
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=resolution_data,
        x="dataset",
        y="megapixels",
    )
    plt.title("Original Image Resolution by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Megapixels")
    plt.tight_layout()

    plt.savefig(
        figures_dir / "04_resolution_distribution.png",
        dpi=160,
    )
    plt.close()


def plot_quality_distribution(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    quality_data = metadata.copy()

    quality_data["quality_display"] = "Official quality unavailable"

    quality_data.loc[
        quality_data["quality_label"] == 1,
        "quality_display",
    ] = "Official: gradable"

    quality_data.loc[
        quality_data["quality_label"] == 0,
        "quality_display",
    ] = "Official: not gradable"

    plt.figure(figsize=(11, 6))
    sns.countplot(
        data=quality_data,
        x="quality_display",
        hue="dataset",
    )
    plt.title("Official Image-Quality Label Availability")
    plt.xlabel("Quality Label")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "05_official_quality_distribution.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=quality_data,
        y="quality_flag_auto",
        order=quality_data["quality_flag_auto"]
        .value_counts()
        .index,
        color="#F59E0B",
    )
    plt.title("Automatic Image-Quality Flags")
    plt.xlabel("Number of Images")
    plt.ylabel("Automatic Quality Flag")
    plt.tight_layout()

    plt.savefig(
        figures_dir / "06_automatic_quality_flags.png",
        dpi=160,
    )
    plt.close()


def plot_lesion_availability(
    metadata: pd.DataFrame,
    figures_dir: Path,
) -> None:
    lesion_summary = (
        metadata.groupby("dataset")
        .agg(
            no_lesion_masks=(
                "lesion_available",
                lambda values: (~values.fillna(False)).sum(),
            ),
            lesion_masks_available=(
                "lesion_available",
                lambda values: values.fillna(False).sum(),
            ),
        )
        .reset_index()
    )

    lesion_plot = lesion_summary.melt(
        id_vars="dataset",
        var_name="annotation_status",
        value_name="images",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=lesion_plot,
        x="dataset",
        y="images",
        hue="annotation_status",
    )
    plt.title("Lesion-Mask Availability by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    plt.savefig(
        figures_dir / "07_lesion_annotation_availability.png",
        dpi=160,
    )
    plt.close()


def plot_sample_images(
    metadata: pd.DataFrame,
    root: Path,
    figures_dir: Path,
    random_seed: int = 42,
) -> None:
    """Save one processed sample image from each dataset."""
    datasets = sorted(metadata["dataset"].dropna().unique())

    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(16, 5),
    )

    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset in zip(axes, datasets):
        candidates = metadata.loc[
            (metadata["dataset"] == dataset)
            & metadata["processed_path"].notna()
        ]

        if candidates.empty:
            axis.set_title(f"{dataset}\nNo processed image")
            axis.axis("off")
            continue

        sample = candidates.sample(
            n=1,
            random_state=random_seed,
        ).iloc[0]

        image_path = root / sample["processed_path"]

        try:
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))

            grade = sample.get("dr_grade_name", "Unlabeled")

            axis.set_title(
                f"{dataset}\n{grade}",
                fontsize=10,
            )

        except OSError:
            axis.set_title(f"{dataset}\nCould not open sample")

        axis.axis("off")

    figure.suptitle("Processed Fundus Image Samples", fontsize=14)
    figure.tight_layout()

    figure.savefig(
        figures_dir / "08_sample_images_by_dataset.png",
        dpi=160,
    )

    plt.close()


def run_phase5_eda(root: Path) -> pd.DataFrame:
    sns.set_theme(style="whitegrid")

    figures_dir, tables_dir = make_output_folders(root)
    metadata = load_metadata(root)

    summary = save_dataset_summary(metadata, tables_dir)

    plot_dr_class_distribution(metadata, figures_dir)
    plot_dr_grade_by_dataset(metadata, figures_dir)
    plot_images_per_dataset(metadata, figures_dir)
    plot_resolution_distribution(metadata, figures_dir)
    plot_quality_distribution(metadata, figures_dir)
    plot_lesion_availability(metadata, figures_dir)
    plot_sample_images(metadata, root, figures_dir)

    print("Phase 5 EDA complete.")
    print(f"Figures saved to: {figures_dir}")
    print(f"Summary table saved to: {tables_dir}")

    return summary