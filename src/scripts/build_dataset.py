import pandas as pd
import random
from sklearn.model_selection import train_test_split

from src.data.parser import load_synthetic_dataset, load_khatt_dataset, load_iam_dataset, load_eval_dataset
from src.config import DATA_PATHS

def main():
    print("🚀 Building Dataset Manifests...")
    
    # ==========================================
    # 1. LOAD TRAINING DATA (Separated by Language)
    # ==========================================
    arabic_train_data = []
    english_train_data = []

    print("\n--- Loading Synthetic ---")
    
    # A1. English Synthetic
    english_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_ENGLISH"]))
    
    # A2. Arabic Synthetic
    arabic_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_ARABIC"]))
    
    # Ignored mixed data as requested
    # all_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_DUAL"]))

    # B. KHATT (Arabic Handwriting)
    print("\n--- Loading KHATT (Arabic) ---")
    arabic_train_data.extend(load_khatt_dataset(
        image_folder=DATA_PATHS["KHATT_IMGS"], 
        label_folder=DATA_PATHS["KHATT_LABELS"], 
        limit=3500
    ))
    
    # C. IAM (English Handwriting)
    print("\n--- Loading IAM (English) ---")
    english_train_data.extend(load_iam_dataset(
        root_dir=DATA_PATHS["IAM_IMGS_ROOT"], 
        labels_path=DATA_PATHS["IAM_LABELS"], 
        limit=6500
    ))

    # ==========================================
    # 2. SPLIT TRAIN / VALIDATION & SAVE (OCR)
    # ==========================================
    print(f"\nTotal Arabic Samples: {len(arabic_train_data)}")
    print(f"Total English Samples: {len(english_train_data)}")
    
    # Process Arabic Data
    df_ar = pd.DataFrame(arabic_train_data)
    df_ar = df_ar.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Process English Data
    df_en = pd.DataFrame(english_train_data)
    df_en = df_en.sample(frac=1, random_state=42).reset_index(drop=True)

    # Use dict.get() with a fallback filename just in case they aren't in config yet
    ar_manifest_path = DATA_PATHS.get("TRAIN_MANIFEST_ARABIC", "train_manifest_ar.csv")
    en_manifest_path = DATA_PATHS.get("TRAIN_MANIFEST_ENGLISH", "train_manifest_en.csv")
    
    df_ar.to_csv(ar_manifest_path, index=False)
    print(f"✅ Saved ARABIC TRAIN manifest: {len(df_ar)} rows -> {ar_manifest_path}")

    df_en.to_csv(en_manifest_path, index=False)
    print(f"✅ Saved ENGLISH TRAIN manifest: {len(df_en)} rows -> {en_manifest_path}")

    # ==========================================
    # 3. BUILD TEST SET (Hold-out)
    # ==========================================
    print("\n--- Loading Evaluation Set ---")
    test_data = load_eval_dataset(DATA_PATHS["EVAL_IMGS"], DATA_PATHS["EVAL_LABELS"])
    
    if test_data:
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(DATA_PATHS.get("TEST_MANIFEST", "test_manifest.csv"), index=False)
        print(f"✅ Saved TEST manifest:  {len(test_df)} rows -> {DATA_PATHS.get('TEST_MANIFEST', 'test_manifest.csv')}")
    else:
        print("⚠️ No evaluation data found.")

    # ==========================================
    # 4. BUILD LANGUAGE CLASSIFIER DATASET
    # ==========================================
    print("\n--- Building Language Classifier Dataset ---")
    
    # Load data freshly here to avoid mutating the OCR dataset objects
    class_khatt = load_khatt_dataset(
        image_folder=DATA_PATHS["KHATT_IMGS"], 
        label_folder=DATA_PATHS["KHATT_LABELS"], 
        limit=10000 # Using high limit to pull all available, we balance below
    )
    
    class_iam = load_iam_dataset(
        root_dir=DATA_PATHS["IAM_IMGS_ROOT"], 
        labels_path=DATA_PATHS["IAM_LABELS"], 
        limit=10000 
    )

    # Add classification labels
    for item in class_khatt:
        item["lang"] = "arabic"
        item["lang_label"] = 0

    for item in class_iam:
        item["lang"] = "english"
        item["lang_label"] = 1

    # Balance the datasets
    random.seed(42)
    random.shuffle(class_khatt)
    random.shuffle(class_iam)

    min_samples = min(len(class_khatt), len(class_iam))
    print(f"Balancing datasets: Found {len(class_khatt)} Arabic and {len(class_iam)} English samples.")
    print(f"Truncating both to {min_samples} samples per class for perfectly balanced data.")

    khatt_balanced = class_khatt[:min_samples]
    iam_balanced = class_iam[:min_samples]

    classifier_train_data = khatt_balanced + iam_balanced

    # Shuffle and save
    df_class = pd.DataFrame(classifier_train_data)
    df_class = df_class.sample(frac=1, random_state=42).reset_index(drop=True)

    class_save_path = DATA_PATHS.get("CLASSIFIER_MANIFEST", "lang_classifier_manifest.csv")
    df_class.to_csv(class_save_path, index=False)
    
    print(f"✅ Saved Language Classifier manifest: {len(df_class)} rows -> {class_save_path}")

    print("\n--- Building Language Classifier Test Set ---")
    
    # Note: You can move these paths into your src.config DATA_PATHS later
    class_test_csv = r"data/processed/cleaned_crop_labels.csv"
    class_test_img_dir = r"data/processed/" # Assuming images are in the same folder
    
    if os.path.exists(class_test_csv):
        df_class_eval = pd.read_csv(class_test_csv)
        classifier_test_data = []
        
        for _, row in df_class_eval.iterrows():
            filename = row['filename']
            label = str(row['label']).strip().title() # standardizes to "Arabic" or "English"
            
            full_path = os.path.join(class_test_img_dir, filename)
            
            if os.path.exists(full_path):
                # Map to match training logic: Arabic = 0, English = 1
                lang_label = 0 if label == "Arabic" else 1
                
                classifier_test_data.append({
                    "file_path": full_path,
                    "lang": label.lower(),
                    "lang_label": lang_label,
                    "source": "classifier_eval"
                })
        
        if classifier_test_data:
            df_class_test_out = pd.DataFrame(classifier_test_data)
            
            # Save the new test manifest
            class_test_save_path = DATA_PATHS.get("CLASSIFIER_TEST_MANIFEST", "lang_classifier_test_manifest.csv")
            df_class_test_out.to_csv(class_test_save_path, index=False)
            
            print(f"✅ Saved Language Classifier TEST manifest: {len(df_class_test_out)} rows -> {class_test_save_path}")
        else:
            print(f"⚠️ No valid images found in {class_test_img_dir}. Check if the image folder path is correct.")
    else:
        print(f"⚠️ Classifier test CSV not found at {class_test_csv}")


if __name__ == "__main__":
    main()