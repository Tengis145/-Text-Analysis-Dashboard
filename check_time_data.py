import pandas as pd
import io

file1 = r"C:\Users\Dell\Downloads\50_analysis (2).xlsx"

print("="*70)
print("CHECKING 50_ANALYSIS FILE")
print("="*70)

xls = pd.ExcelFile(file1, engine='openpyxl')
print(f"Sheets: {xls.sheet_names[:5]}...")

for sheet_name in xls.sheet_names[:2]:
    try:
        df = pd.read_excel(file1, sheet_name=sheet_name, engine='openpyxl')
        print(f"\n{sheet_name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:10]}")
        print(f"  Data sample:\n{df.head(3).to_string()}")
    except Exception as e:
        print(f"  Error: {e}")
