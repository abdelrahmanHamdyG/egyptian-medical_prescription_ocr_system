import pandas as pd
import random
from sklearn.model_selection import train_test_split

from src.data.parser import load_synthetic_dataset, load_khatt_dataset,load_iam_dataset,load_eval_dataset
from src.config import DATA_PATHS

def main():
    print("🚀 Building Dataset Manifests...")
    
    # ==========================================
    # 1. LOAD TRAINING DATA
    # ==========================================
    all_train_data = []

    # A. Synthetic Data (Your generated prescriptions)
    # We load ALL synthetic data because it's high quality and specific to your problem
    print("\n--- Loading Synthetic ---")
    
    all_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_ENGLISH"]))
    all_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_ARABIC"]))
    
    all_train_data.extend(load_synthetic_dataset(DATA_PATHS["SYNTH_DUAL"]))

    print(len(all_train_data))
    print(all_train_data[-1])

    
    # B. KHATT (Arabic Handwriting)
    
    print("\n--- Loading KHATT ---")
    all_train_data.extend(load_khatt_dataset(
        image_folder=DATA_PATHS["KHATT_IMGS"], 
        label_folder=DATA_PATHS["KHATT_LABELS"], 
        limit=3500
    ))
    
    # C. IAM (English Handwriting)
    
    all_train_data.extend(load_iam_dataset(
        root_dir=DATA_PATHS["IAM_IMGS_ROOT"], 
        labels_path=DATA_PATHS["IAM_LABELS"], 
        limit=6500
    ))

    

    # ==========================================
    # 2. SPLIT TRAIN / VALIDATION
    # ==========================================
    print(f"\nTotal Samples Found: {len(all_train_data)}")
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_train_data)
    
    # Shuffle everything before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    

    
    df.to_csv(DATA_PATHS["TRAIN_MANIFEST"], index=False)
    
    
    print(f"✅ Saved TRAIN manifest: {len(df)} rows -> {DATA_PATHS['TRAIN_MANIFEST']}")
    

    # ==========================================
    # 3. BUILD TEST SET (Hold-out)
    # ==========================================
    # This is your specific eval dataset. We NEVER mix this with training data.
    print("\n--- Loading Evaluation Set ---")
    test_data = load_eval_dataset(DATA_PATHS["EVAL_IMGS"], DATA_PATHS["EVAL_LABELS"])
    
    if test_data:
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(DATA_PATHS["TEST_MANIFEST"], index=False)
        print(f"✅ Saved TEST manifest:  {len(test_df)} rows -> {DATA_PATHS['TEST_MANIFEST']}")
    else:
        print("⚠️ No evaluation data found.")

if __name__ == "__main__":
    main()