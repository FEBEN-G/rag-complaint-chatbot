"""
Quick script to inspect the pre-built embeddings parquet file.
"""
import pandas as pd
import pyarrow.parquet as pq

# Load parquet file
parquet_file = 'data/raw/complaint_embeddings.parquet'
print(f"Loading: {parquet_file}\n")

# Read parquet metadata
parquet_table = pq.read_table(parquet_file)
print(f"Total rows: {parquet_table.num_rows:,}")
print(f"Total columns: {parquet_table.num_columns}")
print(f"\nColumn names:")
for col in parquet_table.column_names:
    print(f"  - {col}")

# Load as pandas for inspection
df = pd.read_parquet(parquet_file)
print(f"\nDataFrame shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nColumn info:")
df.info()
