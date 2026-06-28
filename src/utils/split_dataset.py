import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_ROOT = "data/raw/cheating_detection"
OUTPUT_ROOT = "data/splits/cheating"

classes = [
    "Normal",
    "Using mobile",
    "Looking around",
    "Leaning to copy",
    "Sharing answers"
]

for class_name in classes:

    source_folder = os.path.join(DATASET_ROOT, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    train_imgs, temp = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, image_list in splits.items():

        destination = os.path.join(
            OUTPUT_ROOT,
            split_name,
            class_name
        )

        os.makedirs(destination, exist_ok=True)

        for image in image_list:
            shutil.copy(
                os.path.join(source_folder, image),
                os.path.join(destination, image)
            )

    print(
        f"{class_name}: "
        f"Train={len(train_imgs)}, "
        f"Val={len(val_imgs)}, "
        f"Test={len(test_imgs)}"
    )

print("Dataset splitting completed.")