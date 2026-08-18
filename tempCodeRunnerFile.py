
import os
import random
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split


DATASET_DIR = r"C:\Users\khale\.cache\kagglehub\datasets\andrewmvd\face-mask-detection\versions\1"

IMAGES_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

ANNOTATIONS_DIR = os.path.join(
    DATASET_DIR,
    "annotations"
)

OUTPUT_DIR = "processed_dataset"

IMG_SIZE = (224, 224)

CLASSES = [
    "with_mask",
    "without_mask",
    "mask_weared_incorrect"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Target number of training images
# for minority classes.
TARGET_IMAGES = 1500

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1) DATA AUGMENTATION
# ============================================================

augmentation_pipeline = A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.Rotate(
        limit=15,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    ),

    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3
    )
])


# ============================================================
# 2) CHECK IMAGE
# ============================================================

def is_valid_image(image_path):

    try:

        img = Image.open(
            image_path
        )

        img.verify()

        return True

    except Exception:

        return False


# ============================================================
# 3) CLEAN DATASET
# ============================================================

def clean_dataset():

    print(
        "\n========== Cleaning Dataset ==========\n"
    )

    removed = 0

    for filename in os.listdir(
        IMAGES_DIR
    ):

        image_path = os.path.join(
            IMAGES_DIR,
            filename
        )

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            continue

        if not is_valid_image(
            image_path
        ):

            print(
                "Removing corrupted image:",
                filename
            )

            os.remove(
                image_path
            )

            removed += 1

    print(
        f"\nCleaning completed. "
        f"Removed {removed} corrupted images."
    )


# ============================================================
# 4) PARSE XML
# ============================================================

def parse_annotation(xml_path):


    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    faces = []

    for obj in root.findall(
        "object"
    ):

        class_name = obj.find(
            "name"
        ).text

        if class_name not in CLASSES:

            continue

        bounding_box = obj.find(
            "bndbox"
        )

        xmin = int(
            bounding_box.find(
                "xmin"
            ).text
        )

        ymin = int(
            bounding_box.find(
                "ymin"
            ).text
        )

        xmax = int(
            bounding_box.find(
                "xmax"
            ).text
        )

        ymax = int(
            bounding_box.find(
                "ymax"
            ).text
        )

        faces.append({

            "class": class_name,

            "xmin": xmin,

            "ymin": ymin,

            "xmax": xmax,

            "ymax": ymax
        })

    return faces


# ============================================================
# 5) EXTRACT FACES
# ============================================================

def extract_faces():

    print(
        "\n========== Extracting Faces ==========\n"
    )

    os.makedirs(
        "raw_dataset",
        exist_ok=True
    )

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(
                "raw_dataset",
                class_name
            ),
            exist_ok=True
        )

    image_files = [

        f

        for f in os.listdir(
            IMAGES_DIR
        )

        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    total_faces = 0

    for image_filename in image_files:

        image_path = os.path.join(
            IMAGES_DIR,
            image_filename
        )

        xml_filename = (
            os.path.splitext(
                image_filename
            )[0]
            + ".xml"
        )

        xml_path = os.path.join(
            ANNOTATIONS_DIR,
            xml_filename
        )

        if not os.path.exists(
            xml_path
        ):

            print(
                "Warning: XML not found for",
                image_filename
            )

            continue

        image = cv2.imread(
            image_path
        )

        if image is None:

            continue

        faces = parse_annotation(
            xml_path
        )

        for face_index, face in enumerate(
            faces
        ):

            xmin = face["xmin"]
            ymin = face["ymin"]
            xmax = face["xmax"]
            ymax = face["ymax"]

            # Make sure coordinates
            # are inside image

            xmin = max(
                0,
                xmin
            )

            ymin = max(
                0,
                ymin
            )

            xmax = min(
                image.shape[1],
                xmax
            )

            ymax = min(
                image.shape[0],
                ymax
            )

    

            face_image = image[
                ymin:ymax,
                xmin:xmax
            ]

            if face_image.size == 0:

                continue

            class_name = face["class"]

            output_filename = (

                f"{os.path.splitext(image_filename)[0]}"

                f"_face{face_index}.jpg"
            )

            output_path = os.path.join(

                "raw_dataset",

                class_name,

                output_filename
            )

            cv2.imwrite(
                output_path,
                face_image
            )

            total_faces += 1

    print(
        "\nFace extraction completed."
    )

    print(
        f"Total extracted faces: "
        f"{total_faces}"
    )


# ============================================================
# 6) GET CLASS IMAGES
# ============================================================

def get_class_images(class_name):

    class_folder = os.path.join(

        "raw_dataset",

        class_name
    )

    images = [

        f

        for f in os.listdir(
            class_folder
        )

        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    return images


# ============================================================
# 7) CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    for split in [
        "train",
        "val",
        "test"
    ]:

        for class_name in CLASSES:

            folder = os.path.join(

                OUTPUT_DIR,

                split,

                class_name
            )

            os.makedirs(
                folder,
                exist_ok=True
            )


# ============================================================
# 8) SPLIT DATASET
# ============================================================

def split_dataset():

    print(
        "\n========== Splitting Dataset ==========\n"
    )

    create_output_directories()

    summary = {}

    for class_name in CLASSES:

        images = get_class_images(
            class_name
        )

        if len(images) == 0:

            print(
                f"Warning: No images found "
                f"for {class_name}"
            )

            continue

        train_images, temp_images = train_test_split(

            images,

            test_size=(
                VAL_RATIO + TEST_RATIO
            ),

            random_state=RANDOM_SEED
        )


        val_images, test_images = train_test_split(

            temp_images,

            test_size=0.5,

            random_state=RANDOM_SEED
        )

        splits = {

            "train": train_images,

            "val": val_images,

            "test": test_images
        }

        summary[class_name] = {}

        for split_name, split_images in splits.items():

            summary[class_name][split_name] = len(
                split_images
            )

            for filename in split_images:

                source = os.path.join(

                    "raw_dataset",

                    class_name,

                    filename
                )

                destination = os.path.join(

                    OUTPUT_DIR,

                    split_name,

                    class_name,

                    filename
                )

                shutil.copy2(

                    source,

                    destination
                )

    print(
        "\n========== Split Summary ==========\n"
    )

    for class_name, counts in summary.items():

        print(

            f"{class_name}: "

            f"Train={counts['train']} | "

            f"Validation={counts['val']} | "

            f"Test={counts['test']}"
        )

    return summary


# ============================================================
# 9) RESIZE IMAGES
# ============================================================

def resize_dataset():

    print(
        "\n========== Resizing Images ==========\n"
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        for class_name in CLASSES:

            folder = os.path.join(

                OUTPUT_DIR,

                split,

                class_name
            )

            for filename in os.listdir(
                folder
            ):

                image_path = os.path.join(

                    folder,

                    filename
                )

                image = cv2.imread(
                    image_path
                )

                if image is None:

                    continue

                resized = cv2.resize(

                    image,

                    IMG_SIZE,

                    interpolation=cv2.INTER_AREA
                )

                cv2.imwrite(

                    image_path,

                    resized
                )

    print(
        "Resizing completed."
    )


# ============================================================
# 10) AUGMENT TRAINING DATA ONLY
# ============================================================

def augment_training_data():

    print(
        "\n========== Data Augmentation ==========\n"
    )

    train_dir = os.path.join(

        OUTPUT_DIR,

        "train"
    )

    for class_name in CLASSES:

        class_folder = os.path.join(

            train_dir,

            class_name
        )



        images = [

            f

            for f in os.listdir(
                class_folder
            )

            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )

            and "_aug" not in f
        ]

        original_count = len(
            images
        )


        if class_name == "with_mask":

            print(

                f"{class_name}: "

                f"{original_count} original images - "

                f"no additional augmentation"
            )

            continue



        if original_count >= TARGET_IMAGES:

            print(

                f"{class_name}: "

                f"{original_count} images - "

                f"augmentation not required"
            )

            continue

        # ----------------------------------------------------
        # Calculate how many images we need
        # ----------------------------------------------------

        required_images = (

            TARGET_IMAGES

            - original_count
        )

        print(

            f"{class_name}: "

            f"{original_count} original images -> "

            f"creating {required_images} "
            f"augmented images"
        )

        generated = 0

        # ----------------------------------------------------
        # Generate augmented images
        # ----------------------------------------------------

        while generated < required_images:

     

            filename = random.choice(
                images
            )

            image_path = os.path.join(

                class_folder,

                filename
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                continue


            augmented = augmentation_pipeline(

                image=image
            )

            augmented_image = augmented[
                "image"
            ]

            name, extension = os.path.splitext(
                filename
            )

            new_filename = (

                f"{name}_aug{generated}"

                f"{extension}"
            )

            new_path = os.path.join(

                class_folder,

                new_filename
            )

            cv2.imwrite(

                new_path,

                augmented_image
            )

            generated += 1

        print(

            f"Augmentation completed for: "

            f"{class_name}"
        )

    print(
        "\nTraining dataset balancing completed."
    )


# ============================================================
# 11) FINAL DATASET SUMMARY
# ============================================================

def print_final_summary():

    print(
        "\n========== FINAL DATASET ==========\n"
    )

    total = 0

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"\n{split.upper()}:"
        )

        split_total = 0

        for class_name in CLASSES:

            folder = os.path.join(

                OUTPUT_DIR,

                split,

                class_name
            )

            count = len([

                f

                for f in os.listdir(
                    folder
                )

                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ])

            print(

                f"  {class_name}: "
                f"{count}"
            )

            split_total += count

        print(

            f"  Total: "
            f"{split_total}"
        )

        total += split_total

    print(
        f"\nTOTAL IMAGES: {total}"
    )


# ============================================================
# 12) MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n"
        "============================================\n"
        " FACE MASK DETECTION - PREPROCESSING\n"
        "============================================\n"
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(
        DATASET_DIR
    ):

        print(

            f"ERROR: Dataset folder "
            f"'{DATASET_DIR}' was not found."
        )

        return

    if not os.path.exists(
        IMAGES_DIR
    ):

        print(

            f"ERROR: Images folder "
            f"'{IMAGES_DIR}' was not found."
        )

        return

    if not os.path.exists(
        ANNOTATIONS_DIR
    ):

        print(

            f"ERROR: Annotations folder "
            f"'{ANNOTATIONS_DIR}' was not found."
        )

        return

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Delete previous generated datasets.
    #
    # This prevents duplicated images when
    # running the script multiple times.
    # --------------------------------------------------------

    if os.path.exists(
        "raw_dataset"
    ):

        print(
            "\nRemoving old raw_dataset..."
        )

        shutil.rmtree(
            "raw_dataset"
        )

    if os.path.exists(
        OUTPUT_DIR
    ):

        print(
            "Removing old processed_dataset..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    clean_dataset()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    extract_faces()

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    split_dataset()

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    resize_dataset()

    # --------------------------------------------------------
    # Step 5
    #
    # IMPORTANT:
    # augmentation happens AFTER splitting
    # and ONLY on training data.
    # --------------------------------------------------------

    augment_training_data()

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    print_final_summary()

    print(
        "\n============================================"
    )

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY!"
    )

    print(
        f"Processed dataset location: "
        f"{OUTPUT_DIR}/"
    )

    print(
        "============================================\n"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
