import numpy as np
import yaml
import shutil
from pathlib import Path
import pandas as pd


def write_box_files(reference: pd.DataFrame, output_path: Path) -> None:
    for image_name, box in reference.iterrows():
        with Path(output_path, str(image_name)).with_suffix(".txt").open("w") as file:
            file.write("0\t" + "\t".join(str(x) for x in box))


if __name__ == "__main__":
    raw_dataset_path = Path("data", "african_wildlife", "raw")
    processed_dataset_path = Path("data", "african_wildlife", "processed")

    if processed_dataset_path.exists():
        shutil.rmtree(processed_dataset_path)

    processed_images_path = Path(processed_dataset_path, "images")
    processed_labels_path = Path(processed_dataset_path, "labels")

    for path in [processed_images_path, processed_labels_path]:
        path.mkdir(parents=True, exist_ok=True)

    for path in raw_dataset_path.rglob("*.jpg"):
        new_name = f"{path.parent.stem}-{path.name}"
        shutil.copy(path, Path(processed_images_path, new_name))

    for path in raw_dataset_path.rglob("*.txt"):
        new_name = f"{path.parent.stem}-{path.name}"
        shutil.copy(path, Path(processed_labels_path, new_name))

    samples = list(processed_images_path.rglob("*.jpg"))
    assert samples

    train_samples = np.random.choice(
        samples, size=int(round(len(samples) * 0.8)), replace=False
    )
    val_samples = [x for x in samples if x not in train_samples]

    train_reference_path = Path(processed_dataset_path, "train.txt")
    val_reference_path = Path(processed_dataset_path, "val.txt")

    with train_reference_path.open("w") as file:
        file.write("\n".join(str(x.absolute()) for x in train_samples))

    with val_reference_path.open("w") as file:
        file.write("\n".join(str(x.absolute()) for x in val_samples))

    metadata = {
        "path": processed_dataset_path.absolute().__str__(),
        "train": train_reference_path.absolute().__str__(),
        "val": val_reference_path.absolute().__str__(),
        "test": None,
        "names": {
            0: "buffalo",
            1: "elephant",
            2: "rhino",
            3: "zebra",
        },
    }

    with Path(processed_dataset_path, "dataset.yaml").open("w") as file:
        yaml.dump(metadata, file)
