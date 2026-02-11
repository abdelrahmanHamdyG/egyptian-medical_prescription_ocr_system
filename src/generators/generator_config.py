# config.py

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
            "MORPHOLOGY": 0.25,
            "INK_NOISE": 0.20,
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
