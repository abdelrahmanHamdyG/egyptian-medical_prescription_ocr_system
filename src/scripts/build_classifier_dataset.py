import os
import random
import pandas as pd

from src.data.parser import (
    load_synthetic_dataset,
    load_iam_dataset,
    load_khatt_dataset_for_classifier,
    load_raw_image_folder
)
from src.config import DATA_PATHS


def main():
    print("🚀 Building Balanced Language Classifier Dataset (10k total)...")

    random.seed(42)

    classifier_train_data = []

    # ======================================================
    # 1️⃣ ARABIC (5,000)
    # ======================================================
    print("\n--- Loading ARABIC Data ---")

    # Synthetic Arabic (2,500)
    synth_ar_full = load_synthetic_dataset(DATA_PATHS["SYNTH_ARABIC_CLASSIFER"])
    random.shuffle(synth_ar_full)
    synth_ar_sampled = synth_ar_full[:3000]

    # KHATT Crops (2,500)
    khatt_data = load_khatt_dataset_for_classifier(
        image_folder=DATA_PATHS["KHATT_CROPS_IMGS"],
        labels_path=DATA_PATHS["KHATT_CROPS_LABELS"],
        limit=8500
    )
    # muharaf_data=load_raw_image_folder(DATA_PATHS["MUHARAF_IMGS"],'arabic',3000)
    arabic_data = synth_ar_sampled + khatt_data

    for item in arabic_data:
        item["lang"] = "arabic"
        item["lang_label"] = 0

    classifier_train_data.extend(arabic_data)

    print(f"-> Arabic Total: {len(arabic_data)}")


    # ======================================================
    # 2️⃣ ENGLISH (5,000)
    # ======================================================
    print("\n--- Loading ENGLISH Data ---")

    # Synthetic English (2,500)
    synth_en_full = load_synthetic_dataset(DATA_PATHS["SYNTH_ENGLISH_CLASSIFER"])
    random.shuffle(synth_en_full)
    synth_en_sampled = synth_en_full[:3000]

    # IAM Handwritten (2,500)
    iam_data = load_iam_dataset(
        root_dir=DATA_PATHS["IAM_IMGS_ROOT"],
        labels_path=DATA_PATHS["IAM_LABELS"],
        limit=8500
    )
    # parquet_data=load_raw_image_folder(DATA_PATHS["PARQUET_DATA"],"english",3000)

    english_data = synth_en_sampled + iam_data

    for item in english_data:
        item["lang"] = "english"
        item["lang_label"] = 1

    classifier_train_data.extend(english_data)

    print(f"-> English Total: {len(english_data)}")


    # ======================================================
    # 3️⃣ SHUFFLE & SAVE TRAIN MANIFEST
    # ======================================================
    df_train = pd.DataFrame(classifier_train_data)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    train_save_path = DATA_PATHS.get(
        "TRAIN_CLASSIFIER_MANIFEST",
        "lang_classifier_manifest.csv"
    )

    df_train.to_csv(train_save_path, index=False)

    print(f"\n✅ Saved TRAIN manifest: {len(df_train)} rows -> {train_save_path}")
    print(f"Train Distribution:\n{df_train['lang'].value_counts().to_string()}")


    # ======================================================
    # 4️⃣ BUILD LANGUAGE CLASSIFIER TEST SET
    # ======================================================
    print("\n--- Building Language Classifier Test Set ---")

    class_test_csv = DATA_PATHS.get(
        "CLASSIFIER_TEST_LABELS",
        r"data/processed/cleaned_crop_labels.csv"
    )

    class_test_img_dir = DATA_PATHS.get(
        "CLASSIFIER_TEST_IMGS",
        r"data/processed/"
    )

    if os.path.exists(class_test_csv):

        df_class_eval = pd.read_csv(class_test_csv)
        classifier_test_data = []

        for _, row in df_class_eval.iterrows():

            filename = str(row["filename"]).strip()
            label = str(row["label"]).strip().title()  # Arabic / English

            full_path = os.path.join(class_test_img_dir, filename)

            if os.path.exists(full_path):

                lang_label = 0 if label == "Arabic" else 1

                classifier_test_data.append({
                    "file_path": full_path,
                    "lang": label.lower(),
                    "lang_label": lang_label,
                    "source": "classifier_eval"
                })

        if classifier_test_data:

            df_test = pd.DataFrame(classifier_test_data)

            test_save_path = DATA_PATHS.get(
                "TEST_CLASSIFIER_MANIFEST",
                "lang_classifier_test_manifest.csv"
            )

            df_test.to_csv(test_save_path, index=False)

            print(f"✅ Saved TEST manifest: {len(df_test)} rows -> {test_save_path}")
            print(f"Test Distribution:\n{df_test['lang'].value_counts().to_string()}")

        else:
            print(f"⚠️ No valid images found in {class_test_img_dir}. Check image folder path.")

    else:
        print(f"⚠️ Classifier test CSV not found at {class_test_csv}")


if __name__ == "__main__":
    main()