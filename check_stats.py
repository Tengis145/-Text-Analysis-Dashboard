# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

file1 = r"C:\Users\Dell\Downloads\50_analysis (2).xlsx"
xls = pd.ExcelFile(file1)
print(f"File: 50_analysis (2).xlsx")
print(f"Sheets: {len(xls.sheet_names)}")

total_rows = 0
total_cols = 0
for s in xls.sheet_names:
    df = pd.read_excel(file1, sheet_name=s)
    if not df.empty:
        total_rows += len(df)
        total_cols += len(df.columns)
        missing = int(df.isna().sum().sum())
        dup = int(df.duplicated().sum())
        print(f"  {s}: {len(df)} rows, {len(df.columns)} cols, missing={missing}, dup={dup}")

print(f"\nTotal: {total_rows} rows, {total_cols} total cols")
