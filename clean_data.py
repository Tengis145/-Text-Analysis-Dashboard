import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("DATA CLEANING SCRIPT")
print("="*70)

# Load files
file1 = r"C:\Users\Dell\Downloads\50_analysis (2).xlsx"
file2 = r"C:\Users\Dell\Downloads\DATA_time (1) (1).xlsx"

output_dir = r"C:\Users\Dell\Downloads\CLEANED_DATA"
Path(output_dir).mkdir(exist_ok=True)

print("\n1. CLEANING 50_ANALYSIS FILE...")
print("-" * 70)

xls1 = pd.ExcelFile(file1, engine='openpyxl')

with pd.ExcelWriter(f"{output_dir}/50_analysis_CLEANED.xlsx", engine='openpyxl') as writer:
    for sheet_name in xls1.sheet_names:
        df = pd.read_excel(file1, sheet_name=sheet_name, engine='openpyxl')

        print(f"\n{sheet_name}:")
        print(f"  Before: {df.shape[0]} rows × {df.shape[1]} cols")

        # Count issues
        missing_before = df.isna().sum().sum()
        duplicates_before = df.duplicated().sum()

        print(f"  Missing cells: {missing_before}")
        print(f"  Duplicate rows: {duplicates_before}")

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        # For numeric columns: fill with 0
        # For text columns: fill with "N/A"
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("N/A")

        # Count after cleaning
        missing_after = df.isna().sum().sum()
        duplicates_after = df.duplicated().sum()

        print(f"  After: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Missing cells: {missing_after}")
        print(f"  Duplicate rows: {duplicates_after}")
        print(f"  Cleaned successfully")

        # Write to new file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n\n2. CLEANING DATA_TIME FILE...")
print("-" * 70)

xls2 = pd.ExcelFile(file2, engine='openpyxl')

with pd.ExcelWriter(f"{output_dir}/DATA_time_CLEANED.xlsx", engine='openpyxl') as writer:
    for sheet_name in xls2.sheet_names:
        if sheet_name == "1":
            continue

        df = pd.read_excel(file2, sheet_name=sheet_name, engine='openpyxl')

        print(f"\n{sheet_name}:")
        print(f"  Before: {df.shape[0]} rows × {df.shape[1]} cols")

        # Count issues
        missing_before = df.isna().sum().sum()
        duplicates_before = df.duplicated().sum()

        print(f"  Missing cells: {missing_before}")
        print(f"  Duplicate rows: {duplicates_before}")

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("N/A")

        # Count after cleaning
        missing_after = df.isna().sum().sum()
        duplicates_after = df.duplicated().sum()

        print(f"  After: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Missing cells: {missing_after}")
        print(f"  Duplicate rows: {duplicates_after}")
        print(f"  Cleaned successfully")

        # Write to new file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n" + "="*70)
print("CLEANING COMPLETE!")
print("="*70)
print(f"\nCleaned files saved to:")
print(f"  {output_dir}/")
print(f"  50_analysis_CLEANED.xlsx")
print(f"  DATA_time_CLEANED.xlsx")
print(f"\nNext steps:")
print(f"  1. Review the cleaned files")
print(f"  2. Replace original files if satisfied")
print(f"  3. Re-upload to dashboard")
