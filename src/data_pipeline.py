import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from thefuzz import process, fuzz
from tqdm import tqdm

# Configuration
RAW_SPECS_DIR = "data/raw/specs"
RAW_ANTUTU_DIR = "data/raw/antutu"
PROCESSED_DATA_DIR = "data/processed"

SPECS_DATASET = "devgondaliya007/smartphone-specifications-dataset"
ANTUTU_DATASET = "ireddragonicy/antutu-benchmark"

def setup_directories():
    os.makedirs(RAW_SPECS_DIR, exist_ok=True)
    os.makedirs(RAW_ANTUTU_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_datasets():
    api = KaggleApi()
    api.authenticate()
    if not os.path.exists(os.path.join(RAW_SPECS_DIR, "cleaned_data.csv")):
        print(f"[-] Downloading {SPECS_DATASET}...")
        api.dataset_download_files(SPECS_DATASET, path=RAW_SPECS_DIR, unzip=True)
    
    if not os.path.exists(os.path.join(RAW_ANTUTU_DIR, "Android_AI_General.csv")):
        print(f"[-] Downloading {ANTUTU_DATASET}...")
        api.dataset_download_files(ANTUTU_DATASET, path=RAW_ANTUTU_DIR, unzip=True)

def get_first_csv(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files: raise FileNotFoundError(f"No CSV in {folder_path}")
    return os.path.join(folder_path, files[0])

# --- TEXT PRETTIFIER ---
def prettify_name(name):
    """Mengubah 'samsung s23 fe 5g' menjadi 'Samsung S23 FE 5G'"""
    if pd.isna(name): return ""
    
    name = str(name).title() 
    
    replacements = {
        " 5g": " 5G", " 4g": " 4G", " Fe": " FE", " Gt": " GT",
        " Lte": " LTE", " Ios": " iOS", " Iphon": " iPhone",
        "Poco": "POCO", "Vivo": "vivo", "Iqoo": "iQOO"
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name.strip()

def clean_currency(price_val):
    if pd.isna(price_val): return 0.0
    str_val = str(price_val).strip()
    # Deteksi India Rupee
    if '₹' in str_val:
        clean = str_val.replace('₹', '').replace(',', '')
        try: return float(clean) * 0.012
        except: return 0.0
    # Deteksi Rupiah
    if 'Rp' in str_val:
        clean = str_val.replace('Rp', '').replace('.', '').replace(',', '')
        try: return float(clean) / 15500 
        except: return 0.0
    # Deteksi Dollar/Angka biasa
    clean = str_val.replace('$', '').replace(',', '')
    try:
        val = float(clean)
        # Fallback: Jika angka > 5000 tanpa simbol, anggap Rupee
        if val > 5000: return val * 0.012 
        return val
    except ValueError: return 0.0

def clean_ram(ram_val):
    if pd.isna(ram_val): return 0
    if isinstance(ram_val, (int, float)): return int(ram_val)
    clean_str = str(ram_val).upper().replace('GB', '').replace('RAM', '').strip()
    try: return int(float(clean_str.split()[0]))
    except ValueError: return 0

def clean_storage(storage_val):
    if pd.isna(storage_val): return 0
    if isinstance(storage_val, (int, float)): return int(storage_val)
    clean_str = str(storage_val).upper().replace('GB', '').replace('STORAGE', '').strip()
    try:
        val = int(float(clean_str.split()[0]))
        # Safety: Storage < 10GB kemungkinan error/RAM
        if val < 10: return 0 
        return val
    except ValueError: return 0

def find_best_match(name, choices):
    if pd.isna(name): return None
    # Matching menggunakan lowercase agar akurasi tinggi
    match, score = process.extractOne(name, choices, scorer=fuzz.token_set_ratio)
    if score >= 60: return match
    return None

def process_pipeline():
    setup_directories()
    download_datasets()

    print("[-] detecting CSVs...")
    specs_path = get_first_csv(RAW_SPECS_DIR)
    antutu_path = get_first_csv(RAW_ANTUTU_DIR)
    
    df_specs = pd.read_csv(specs_path)
    df_antutu = pd.read_csv(antutu_path)

    print("[-] Cleaning Data...")
    
    # 1. RAM Logic
    if 'ram_gb' in df_specs.columns: ram_col = 'ram_gb'
    elif 'ram' in df_specs.columns: ram_col = 'ram'
    else: ram_col = [c for c in df_specs.columns if 'ram' in c.lower()][0]
    df_specs['clean_ram'] = df_specs[ram_col].apply(clean_ram)

    # 2. Storage Logic
    if 'storage_gb' in df_specs.columns: storage_col = 'storage_gb'
    elif 'storage' in df_specs.columns: storage_col = 'storage'
    elif 'rom' in df_specs.columns: storage_col = 'rom'
    else: storage_col = None
    
    if storage_col:
        df_specs['clean_storage'] = df_specs[storage_col].apply(clean_storage)
    else:
        df_specs['clean_storage'] = 0

    # 3. Price Logic
    if 'price' in df_specs.columns: price_col = 'price'
    elif 'clean_price' in df_specs.columns: price_col = 'clean_price'
    else: price_col = None
    
    if price_col:
        df_specs['clean_price'] = df_specs[price_col].apply(clean_currency)
    else:
        df_specs['clean_price'] = 0.0

    # 4. Name Logic
    if 'brand' in df_specs.columns and 'model' in df_specs.columns:
        df_specs['full_name'] = df_specs['brand'].astype(str) + " " + df_specs['model'].astype(str)
    elif 'model' in df_specs.columns:
        df_specs['full_name'] = df_specs['model'].astype(str)
    
    # 5. Antutu Logic
    df_antutu.columns = [c.strip() for c in df_antutu.columns]
    if 'Device' in df_antutu.columns: dev_col = 'Device'
    elif 'Model' in df_antutu.columns: dev_col = 'Model'
    else: dev_col = df_antutu.select_dtypes(include=['object']).columns[0]
    
    antutu_devices = df_antutu[dev_col].dropna().tolist()

    print("[-] Merging Datasets...")
    tqdm.pandas()
    
    # MATCHING DULU SEBELUM PRETTIFY (Supaya akurasi fuzzy tetap tinggi)
    df_specs['matched_device'] = df_specs['full_name'].progress_apply(lambda x: find_best_match(x, antutu_devices))
    df_merged = pd.merge(df_specs, df_antutu, left_on='matched_device', right_on=dev_col, how='inner')
    
    # PRETTIFY SETELAH MERGE
    df_merged['full_name'] = df_merged['full_name'].apply(prettify_name)

    # Score Logic
    score_col = None
    for name in ['Score', 'Antutu Score', 'Total Score']:
        if name in df_merged.columns: 
            score_col = name; break
    if not score_col: score_col = [c for c in df_merged.columns if 'Score' in c][0]

    final_df = df_merged[['full_name', 'clean_ram', 'clean_storage', 'clean_price', score_col]]
    final_df.rename(columns={score_col: 'antutu_score', 'clean_storage': 'storage'}, inplace=True)
    
    # FILTER DIPERLONGGAR:
    # Hanya buang yang storage-nya 0. Harga 0 BOLEH MASUK.
    final_df = final_df[final_df['storage'] > 0]
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"[+] Pipeline Done. Total Valid Data: {len(final_df)}")
    if len(final_df) > 0:
        sample = final_df.iloc[0]
        print(f"    Sample: {sample['full_name']} | RAM:{sample['clean_ram']} | ROM:{sample['storage']} | ${sample['clean_price']:.2f}")

if __name__ == "__main__":
    process_pipeline()