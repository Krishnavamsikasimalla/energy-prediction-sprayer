# PHASE 1 — Load & Clean CSVs
import pandas as pd
import glob
import os

RAW_DATA_PATH = '../data/*.csv'
OUTPUT_PATH = '../data/cleaned.csv'

def load_and_clean():
    files = glob.glob(RAW_DATA_PATH)
    print(f"Found {len(files)} CSV files")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before cleaning: {len(df)}")
    df.dropna(inplace=True)
    df = df[df['V'] > 0]
    df = df[df['I'] >= 0]
    df = df[df['Speed'] >= 0]
    print(f"Total rows after cleaning: {len(df)}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    return df

if __name__ == '__main__':
    load_and_clean()
