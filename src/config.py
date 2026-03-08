# config.py
import os 
GENERATOR_CONFIG = {
    # ==========================
    # ENGLISH SYNTHETIC DATASET
    # ==========================
    "ENGLISH": {
        "OUTPUT_DIR": "data/synthetic/eng_only",
        "FONTS_DIR": "data/raw/fonts/english_fonts",
        "FINAL_SIZE": (600, 80),
        "BASE_FONT_SIZE": 58,
        "SAMPLE_SIZE": 6500,
        "PROBS": {
            "ROTATE": 0.5,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
            "CHAR_DROPOUT": 0.1,
        },
        "WORD_COUNTS": [2, 1, "all"],
        "WORD_WEIGHTS": [0.80, 0.15, 0.05],
        
    },
    "ENGLISH_CLASS": {
        "OUTPUT_DIR": "data/synthetic/eng_only_class",
        "FONTS_DIR": "data/raw/fonts/english_fonts",
        "FINAL_SIZE": (384, 128),
        "BASE_FONT_SIZE": 58,
        "SAMPLE_SIZE": 6500,
        "PROBS": {
            "ROTATE": 0.5,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
            "CHAR_DROPOUT": 0.1,
        },
        "WORD_COUNTS": [1, 2, "all"],
        "WORD_WEIGHTS": [0.85, 0.1, 0.05],
        
    },

    # ==========================
    # ARABIC SYNTHETIC DATASET
    # ==========================
    "ARABIC": {
        "CSV_FILE": "data/raw/lexicons/egyptian_medical_instructions.csv",
        "OUTPUT_DIR": "data/synthetic/ara_only",
        "FINAL_SIZE": (500, 100),
        "BASE_FONT_SIZE": 65,
        "SAMPLE_SIZE": 3500,
        "PROBS": {
            "ROTATE": 0.6,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.9,
            "INK_NOISE": 0.9,
            "WAVE_WARP": 0.05,
        },
        
    },
     "ARABIC_CLASS": {
        "CSV_FILE": "data/raw/lexicons/egyptian_medical_instructions.csv",
        "OUTPUT_DIR": "data/synthetic/ara_only_class",
        "FINAL_SIZE": (384,124),
        "BASE_FONT_SIZE": 65,
        "SAMPLE_SIZE": 3500,
        "PROBS": {
            "ROTATE": 0.6,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.9,
            "INK_NOISE": 0.9,
            "WAVE_WARP": 0.05,
        },
        
    },

    # ==========================
    # DUAL-LANGUAGE SYNTHETIC DATASET
    # ==========================
    "DUAL_LANG": {
        "ENG_CSV": "data/raw/lexicons/output.csv",
        "ARA_CSV": "data/raw/lexicons/arabic_instructions.csv",
        "OUTPUT_DIR": "data/synthetic/dual_lang",
        
        "FINAL_SIZE": (800, 160),
        "SAMPLE_SIZE": 3500,
        "ENG_BASE_SIZE": 52,
        "ARA_BASE_SIZE": 48,
        "PROBS": {
            "ROTATE": 0.6,
            "BLUR": 0.2,
            "SALT_PEPPER": 0.15,
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
            "ENG_CHAR_DROP": 0.1,
            "LAYOUT_HORIZONTAL": 0.4,
        },
        "WORD_COUNTS": [2, 1, "all"],
        "WORD_WEIGHTS": [0.80, 0.15, 0.05],
        
    },

    # ==========================
    # GLOBAL SETTINGS
    # ==========================
    "GLOBAL": {
        "MEDICINES":"data/raw/lexicons/medicine_names.csv",
        "INSTRUCTIONS":"data/raw/lexicons/egyptian_medical_instructions.csv",
        "ENG_FONTS_DIR": "data/raw/fonts/english_fonts",
        "ARA_FONTS_DIR": "data/raw/fonts/arabic_fonts",
        "DATA_ROOT": "data/",
        "SYNTH_ROOT": "data/synthetic/",
        "RAW_ROOT": "data/raw/",
        "TEXT_COLORS": [
            (0, 0, 90),
            (20, 20, 20),
            (130, 0, 0),
        ],
        "SEED": 42

    }
}


DATA_PATHS = {
    # --- Synthetic Datasets ---
    # Adjust these folder names if your synthetic output names are different
    "SYNTH_DUAL": os.path.join( "data", "synthetic", "dual_lang"),
    "SYNTH_ARABIC": os.path.join( "data", "synthetic", "ara_only"),
    "SYNTH_ENGLISH": os.path.join( "data", "synthetic", "eng_only"),
    "SYNTH_ARABIC_CLASSIFER": os.path.join( "data", "synthetic", "ara_only_class"),
    "SYNTH_ENGLISH_CLASSIFER": os.path.join( "data", "synthetic", "eng_only_class"),
    
    # --- KHATT (Arabic Handwriting) ---
    # UPDATED: Pointing to separate 'images' and 'labels' folders
    "KHATT_IMGS": os.path.join("data", "raw","real_dataset", "khatt", "images"),
    "MUHARAF_IMGS": os.path.join("data", "raw","real_dataset", "muharaf","public_new"),
    "PARQUET_DATA":os.path.join("data", "raw","real_dataset", "parquet_eng"),
    "KHATT_CROPS_IMGS": os.path.join("data", "raw","real_dataset", "khatt_words"),
    
    "KHATT_LABELS": os.path.join( "data", "raw", "real_dataset","khatt", "labels"),
    "KHATT_CROPS_LABELS": os.path.join( "data", "raw", "real_dataset","khatt_words", "labels.csv"),
    
    
    # --- IAM (English Handwriting) ---
    "IAM_IMGS_ROOT": os.path.join( "data", "raw","real_dataset" ,"iam","iam_words", "words"),
    "IAM_LABELS": os.path.join(  "data", "raw","real_dataset", "iam","iam_words" ,"words.txt"),

    # --- Manual Evaluation Set (Real World Data) ---
    # Path to: data/raw/eval/data
    "EVAL_IMGS": os.path.join(  "data", "raw", "eval", "data"),
    "EVAL_LABELS": os.path.join(  "data", "raw", "eval", "data", "labels.csv"),
    "CLASSIFIER_TEST_IMGS":os.path.join("data","raw","real_dataset","prescription","crops"),
    "CLASSIFIER_TEST_LABELS":os.path.join("data","raw","real_dataset","prescription","cleaned_crop_labels.csv"),
    

    # --- Output Manifests (Generated by build_dataset.py) ---
    "TRAIN_MANIFEST_ARABIC": os.path.join(  "data", "processed", "ARABIC","train.csv"),
    "TRAIN_MANIFEST_ENGLISH": os.path.join(  "data", "processed","ENGLISH", "train.csv"),
    "TRAIN_CLASSIFIER_MANIFEST": os.path.join(  "data", "processed","CLASSIFIER", "train.csv"),
    "TEST_CLASSIFIER_MANIFEST": os.path.join(  "data", "processed","CLASSIFIER", "test.csv"),
    # "VAL_MANIFEST": os.path.join( a "data", "processed", "val.csv"),
    "TEST_MANIFEST": os.path.join( "data", "processed", "test.csv"),
}