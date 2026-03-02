# PHASE 2 — Physics Calculations
import pandas as pd
import numpy as np

INPUT_PATH = '../data/cleaned.csv'
OUTPUT_PATH = '../data/physics.csv'
SPRAYER_WIDTH_M = 1.5
DT = 1  # seconds between readings

def calculate_physics():
    df = pd.read_csv(INPUT_PATH)
    df['Power_W'] = df['V'] * df['I']
    df['Energy_Wh'] = np.cumsum(df['Power_W'] * DT) / 3600
    df['Distance_m'] = np.cumsum(df['Speed'] * DT)
    df['Area_m2'] = df['Distance_m'] * SPRAYER_WIDTH_M
    df['Area_acres'] = df['Area_m2'] / 4047
    df['Spray_Ratio'] = df['Spray_Status'].rolling(300, min_periods=1).mean()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Physics done. Saved to {OUTPUT_PATH}")
    return df

if __name__ == '__main__':
    calculate_physics()
