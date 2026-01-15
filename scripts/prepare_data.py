"""
Data Preparation for Undergraduate Drought Prediction Project
============================================================
Purpose: Prepare simplified dataset with single location and 3 features
- Filters large USDM dataset for one location
- Selects 3 climate features: Precipitation (apcp), Temperature (tsoil), Humidity (lai)
- Creates weekly time-series data for LSTM training
- Handles missing values and normalization
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def prepare_lstm_data(input_csv='data/USDMData.csv', 
                      output_csv='data/LSTM_data_single_location.csv',
                      selected_location='C1',
                      features=['apcp', 'tsoil', 'lai'],
                      target='drought'):
    """
    Prepare data for LSTM model from raw USDM dataset.
    
    Args:
        input_csv: Path to raw USDM data
        output_csv: Path to save processed data
        selected_location: Single grid location to use (e.g., 'C1')
        features: List of 3 climate features to use
        target: Target variable (drought index)
    
    Returns:
        Processed DataFrame with selected features
    """
    
    print("=" * 70)
    print("DATA PREPARATION FOR LSTM DROUGHT PREDICTION")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        return None
    
    # Load data
    print("\n[Step 1] Loading raw data...")
    try:
        df = pd.read_csv(input_csv)
        print(f"   ✓ Loaded {len(df)} total records")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None
    
    # Filter for single location
    print(f"\n[Step 2] Filtering for single location: {selected_location}")
    df_location = df[df['grid'] == selected_location].copy()
    print(f"   ✓ Found {len(df_location)} records for {selected_location}")
    
    if len(df_location) == 0:
        print(f"   ❌ No data found for location {selected_location}")
        print(f"   Available locations: {df['grid'].unique()[:10]}")
        return None
    
    # Select features
    print(f"\n[Step 3] Selecting 3 climate features + target")
    selected_cols = ['time'] + features + [target]
    available_cols = [col for col in selected_cols if col in df_location.columns]
    
    if len(available_cols) < len(selected_cols):
        missing = set(selected_cols) - set(available_cols)
        print(f"   ⚠ Warning: Missing columns {missing}")
        print(f"   Available columns: {df_location.columns.tolist()}")
    
    df_features = df_location[available_cols].copy()
    print(f"   ✓ Selected features: {available_cols[1:]}")
    print(f"   ✓ Data shape: {df_features.shape}")
    
    # Handle missing values
    print(f"\n[Step 4] Handling missing values")
    print(f"   Missing values before:\n{df_features.isnull().sum()}")
    
    # Forward fill then backward fill
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    print(f"   ✓ Applied forward/backward fill")
    print(f"   Missing values after:\n{df_features.isnull().sum()}")
    
    # Sort by time
    print(f"\n[Step 5] Sorting by time (ensure temporal order)")
    df_features = df_features.sort_values('time').reset_index(drop=True)
    print(f"   ✓ Time range: {df_features['time'].min()} to {df_features['time'].max()}")
    
    # Remove time column for numerical processing
    time_col = df_features['time'].copy()
    df_numeric = df_features.drop('time', axis=1).copy()
    
    # Normalize features to [0, 1]
    print(f"\n[Step 6] Normalizing features to [0, 1]")
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns
    )
    
    # Add time back
    df_normalized['time'] = time_col.values
    
    # Reorder columns
    df_normalized = df_normalized[['time'] + features + [target]]
    
    print(f"   ✓ Normalization complete")
    print(f"   Feature ranges: {df_normalized[features].min().to_dict()}")
    
    # Save processed data
    print(f"\n[Step 7] Saving processed data")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', 
                exist_ok=True)
    df_normalized.to_csv(output_csv, index=False)
    print(f"   ✓ Saved to: {output_csv}")
    print(f"   ✓ Final shape: {df_normalized.shape}")
    
    # Summary statistics
    print(f"\n[Summary] Data Preparation Complete")
    print(f"   Location: {selected_location}")
    print(f"   Features: {', '.join(features)}")
    print(f"   Target: {target}")
    print(f"   Records: {len(df_normalized)}")
    print(f"   Time period: {df_normalized['time'].min()} - {df_normalized['time'].max()}")
    
    return df_normalized

if __name__ == "__main__":
    # Run data preparation
    df = prepare_lstm_data(
        input_csv='data/USDMData.csv',
        output_csv='data/LSTM_data_single_location.csv',
        selected_location='C1',
        features=['apcp', 'tsoil', 'lai'],
        target='drought'
    )
    
    if df is not None:
        print(f"\n{'='*70}")
        print("First few rows of processed data:")
        print(df.head(10))
