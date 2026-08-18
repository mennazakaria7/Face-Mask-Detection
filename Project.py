import os
import random
import shutil
import warnings
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = r"C:\Users\khale\.cache\kagglehub\datasets\andrewmvd\face-mask-detection\versions\1"

OUTPUT_DIR = "processed_dataset"
RAW_DIR = "raw_dataset"

IMG_SIZE = (224, 224)

CLASSES = [
    "with_mask",
    "without_mask",
    "mask_weared_incorrect"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_IMAGES = 1500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", message="Error fetching version info")


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    )
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("""
============================================
 FACE MASK DETECTION - PREPROCESSING
============================================
""")

    images_dir = os.path.join(DATASET_DIR, "images")
    annotations_dir = os.path.join(DATASET_DIR, "annotations")

    # --------------------------------------------------------
    # Remove old data
    # --------------------------------------------------------

    if os.path.exists(RAW_DIR):
        print("\nRemoving old raw_dataset...")
        shutil.rmtree(RAW_DIR)

    if os.path.exists(OUTPUT_DIR):
        print("Removing old processed_dataset...")
        shutil.rmtree(OUTPUT_DIR)

    # ========================================================
    # 1) CLEAN DATASET
    # ========================================================

    print("\n========== Cleaning Dataset ==========\n")

    removed = 0

    for file in os.listdir(images_dir):

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(images_dir, file)

        try:
            img = cv2.imread(path)

            if img is None:
                os.remove(path)
                removed += 1

        except Exception:
            os.remove(path)
            removed += 1

    print(
        f"Cleaning completed. "
        f"Removed {removed} corrupted images."
    )

    # ========================================================
    # 2) EXTRACT FACES
    # ========================================================

    print("\n========== Extracting Faces ==========\n")

    for cls in CLASSES:
        os.makedirs(
            os.path.join(RAW_DIR, cls),
            exist_ok=True
        )

    total_faces = 0

    for file in os.listdir(images_dir):

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            images_dir,
            file
        )

        xml_path = os.path.join(
            annotations_dir,
            os.path.splitext(file)[0] + ".xml"
        )

        if not os.path.exists(xml_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        root = ET.parse(xml_path).getroot()

        for i, obj in enumerate(root.findall("object")):

            cls = obj.find("name").text

            if cls not in CLASSES:
                continue

            box = obj.find("bndbox")

            x1 = max(
                0,
                int(box.find("xmin").text)
            )

            y1 = max(
                0,
                int(box.find("ymin").text)
            )

            x2 = min(
                image.shape[1],
                int(box.find("xmax").text)
            )

            y2 = min(
                image.shape[0],
                int(box.find("ymax").text)
            )

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            name = (
                f"{os.path.splitext(file)[0]}"
                f"_face{i}.jpg"
            )

            cv2.imwrite(  os.path.join(  RAW_DIR, cls,  name ), face  )
               
            total_faces += 1

    print("\nFace extraction completed.")
    print(f"Total extracted faces: {total_faces}")

    # ========================================================
    # 3) SPLIT DATASET
    # ========================================================

    print("\n========== Splitting Dataset ==========\n")

    summary = {}

    for cls in CLASSES:

        files = os.listdir( os.path.join(RAW_DIR, cls))
           
        train, temp = train_test_split(
            files,
            test_size=VAL_RATIO + TEST_RATIO,
            random_state=RANDOM_SEED
        )

        val, test = train_test_split(   temp,test_size=0.5, random_state=RANDOM_SEED )
        
        summary[cls] = {  "train": len(train), "val": len(val),"test": len(test)}
        
        for split, split_files in [
            ("train", train),
            ("val", val),
            ("test", test)
        ]:

            folder = os.path.join( OUTPUT_DIR,  split,  cls )
               
            os.makedirs(folder, exist_ok=True)

            for file in split_files:

                image = cv2.imread(
                    os.path.join(
                        RAW_DIR,
                        cls,
                        file
                    )
                )

                if image is None:
                    continue

                image = cv2.resize(  image,IMG_SIZE )
                  
                cv2.imwrite( os.path.join(folder, file),image)
                   
    print("========== Split Summary ==========\n")

    for cls in CLASSES:

        print(
            f"{cls}: "
            f"Train={summary[cls]['train']} | "
            f"Validation={summary[cls]['val']} | "
            f"Test={summary[cls]['test']}"
        )

    # ========================================================
    # 4) RESIZING
    # ========================================================

    print("\n========== Resizing Images ==========\n")
    print("Resizing completed.")

    # ========================================================
    # 5) AUGMENTATION
    # ========================================================

    print("\n========== Data Augmentation ==========\n")

    for cls in CLASSES:

        folder = os.path.join(
            OUTPUT_DIR,
            "train",
            cls
        )

        files = os.listdir(folder)
        original_count = len(files)

        if cls == "with_mask":

            print(
                f"{cls}: {original_count} original images "
                f"- no additional augmentation"
            )

            continue

        if original_count >= TARGET_IMAGES:

            print(
                f"{cls}: {original_count} images "
                f"- augmentation not required"
            )

            continue

        required = TARGET_IMAGES - original_count

        print(
            f"{cls}: {original_count} original images "
            f"-> creating {required} augmented images"
        )

        for i in range(required):

            file = random.choice(files)

            image = cv2.imread(
                os.path.join(folder, file)
            )

            augmented = augmentation(
                image=image
            )["image"]

            cv2.imwrite(
                os.path.join(
                    folder,
                    f"aug_{i}.jpg"
                ),
                augmented
            )

        print(
            f"Augmentation completed for: {cls}"
        )

    print("\nTraining dataset balancing completed.")

    # ========================================================
    # 6) FINAL SUMMARY
    # ========================================================

    print("\n========== FINAL DATASET ==========\n")

    final_counts = {}

    total = 0

    for split in ["train", "val", "test"]:

        print(f"\n{split.upper()}:")

        final_counts[split] = {}

        split_total = 0

        for cls in CLASSES:

            folder = os.path.join(
                OUTPUT_DIR,
                split,
                cls
            )

            count = len(os.listdir(folder))

            final_counts[split][cls] = count

            print(
                f"  {cls}: {count}"
            )

            split_total += count

        print(f"  Total: {split_total}")

        total += split_total

    print(f"\nTOTAL IMAGES: {total}")

    # ========================================================
    # 7) VISUALIZATIONS
    # ========================================================

    create_visualizations(final_counts)

    print("""
============================================
PREPROCESSING COMPLETED SUCCESSFULLY!
============================================
""")

    print(
        f"Processed dataset location: "
        f"{OUTPUT_DIR}/"
    )

    print("""
Visualizations saved inside processed_dataset/
============================================
""")


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(counts):

    print("\n========== Creating Visualizations ==========\n")

    # --------------------------------------------------------
    # 1) Dataset Distribution
    # --------------------------------------------------------

    x = range(len(CLASSES))

    plt.figure(figsize=(10, 6))

    for i, split in enumerate(
        ["train", "val", "test"]
    ):

        values = [
            counts[split][cls]
            for cls in CLASSES
        ]

        positions = [
            p + (i - 1) * 0.25
            for p in x
        ]

        plt.bar( positions, values,width=0.25,  label=split.upper()  )
           
    plt.xticks(  list(x), CLASSES,rotation=20 )
      
    plt.ylabel("Number of Images")
    plt.title("Dataset Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(  os.path.join(OUTPUT_DIR, "dataset_distribution.png" ), dpi=150 )
      
    plt.close()

    # --------------------------------------------------------
    # 2) Final Training Distribution
    # --------------------------------------------------------

    values = [
        counts["train"][cls]
        for cls in CLASSES
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        CLASSES,
        values
    )

    plt.ylabel("Number of Images")
    plt.title("Final Training Class Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_distribution.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 3) Sample Images
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    index = 1

    for cls in CLASSES:

        folder = os.path.join(OUTPUT_DIR, "train",  cls )
            
        files = os.listdir(folder)[:3]

        for file in files:

            image = cv2.imread(
                os.path.join(folder, file)
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            plt.subplot(3, 3, index   )
                
            plt.imshow(image)
            plt.title(cls)
            plt.axis("off")
            index += 1   
               
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "sample_images.png"
        ),
        dpi=150
    )

    plt.close()

    print("Visualizations created:")
    print("  - dataset_distribution.png")
    print("  - training_distribution.png")
    print("  - sample_images.png")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()