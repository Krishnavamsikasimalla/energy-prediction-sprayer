# PHASE 3 — 5-Minute Segmentation
import pandas as pd

INPUT_PATH = '../data/physics.csv'
OUTPUT_PATH = '../data/segments.csv'
SEGMENT_SIZE = 300  # 300 rows = 5 minutes

def create_segments():
    df = pd.read_csv(INPUT_PATH)
    segments = []
    for i in range(0, len(df) - SEGMENT_SIZE, SEGMENT_SIZE):
        chunk = df.iloc[i:i + SEGMENT_SIZE]
        row = {
            'Avg_Speed':       chunk['Speed'].mean(),
            'Avg_Gradient':    chunk['Gradient'].mean(),
            'Spray_Ratio':     chunk['Spray_Status'].mean(),
            'Area_acres':      chunk['Area_acres'].iloc[-1] - chunk['Area_acres'].iloc[0],
            'Total_Energy_Wh': chunk['Energy_Wh'].iloc[-1] - chunk['Energy_Wh'].iloc[0],
            'Source':          chunk['source_file'].iloc[0]
        }
        segments.append(row)
    seg_df = pd.DataFrame(segments)
    seg_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(seg_df)} segments. Saved to {OUTPUT_PATH}")
    return seg_df

if __name__ == '__main__':
    create_segments()
