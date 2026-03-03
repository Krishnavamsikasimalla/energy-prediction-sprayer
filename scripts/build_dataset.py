#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pandas as pd

# Updated absolute imports pointing to your new package
from sprayer_energy.decoders.mf4_reader import extract_can_frames
from sprayer_energy.decoders.dbc_decoder import decode_dataframe
from sprayer_energy.data_pipeline.windowing import split_into_windows

# Set up standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_file(path: Path) -> pd.DataFrame | None:
    logger.info(f"Processing {path.name}...")

    try:
        # 1. Extract raw CAN frames
        raw_df = extract_can_frames(path)

        # 2. Decode signals using DBC
        decoded_df = decode_dataframe(raw_df)

        if decoded_df is None or decoded_df.empty:
            logger.warning(f"No decoded signals found in {path.name}.")
            return None

        # 3. Split into time windows (30 sec windows)
        windowed_df = split_into_windows(decoded_df)

        if windowed_df is None or windowed_df.empty:
            logger.warning(f"No valid windows generated for {path.name}.")
            return None

        # 4. Add file reference for traceability
        windowed_df["Source_File"] = path.name

        return windowed_df
        
    except Exception as e:
        logger.error(f"Critical error processing {path.name}: {e}")
        return None

def main(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    # Dynamically find all mf4 files
    files = sorted(input_path.glob("*.mf4"))
    
    if not files:
        logger.warning(f"No .mf4 files found in {input_path}")
        return

    logger.info(f"Found {len(files)} files. Starting pipeline...")
    all_windows = []

    for f in files:
        result = process_file(f)
        if result is not None:
            all_windows.append(result)

    if not all_windows:
        logger.error("Pipeline finished, but no valid data was generated.")
        return

    # Merge all window DataFrames
    logger.info("Merging datasets...")
    df = pd.concat(all_windows, ignore_index=True)

    # Ensure output directory exists and save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Dataset successfully saved to {output_path}")
    logger.info(f"Total training rows: {len(df)}")
    
    print("\nDataset Preview:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw CAN MF4 files into a training dataset.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory containing raw .mf4 files")
    parser.add_argument("--output_file", type=str, default="data/processed/training_dataset.csv", help="Path to save processed data")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_file)