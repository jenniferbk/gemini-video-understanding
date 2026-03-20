import pandas as pd
import sys

print("=== ANALYZING EXCEL FILES ===\n")

files = {
    'annaB': '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx',
    'Uyi': '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx',
    'combined': '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'
}

for name, filepath in files.items():
    print(f"\n{'='*60}")
    print(f"FILE: {name}")
    print('='*60)

    try:
        xl_file = pd.ExcelFile(filepath)
        print(f"Sheets: {xl_file.sheet_names}")

        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            print(f"\n  Sheet: '{sheet_name}'")
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)}")

            # Show first few rows
            if len(df) > 0:
                print(f"\n  First few rows:")
                print(df.head(3).to_string(index=False))
            else:
                print("  (Empty sheet)")

    except Exception as e:
        print(f"Error reading {name}: {e}")
