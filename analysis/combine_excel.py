import pandas as pd
import sys
from pathlib import Path

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

print("=== ANALYZING COMBINED FILE ===\n")

# Load combined file to see what user has done
combined_xl = pd.ExcelFile(combined_path)
print(f"Combined file has these sheets: {combined_xl.sheet_names}\n")

# Look at one of the manually combined sheets to understand the format
for sheet in combined_xl.sheet_names[:3]:
    df = pd.read_excel(combined_path, sheet_name=sheet)
    print(f"\nSheet: '{sheet}'")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\n" + "="*80)

# Now let's see what sheets we need to add (from both source files)
annaB_xl = pd.ExcelFile(annaB_path)
uyi_xl = pd.ExcelFile(uyi_path)

annaB_sheets = set(annaB_xl.sheet_names)
uyi_sheets = set(uyi_xl.sheet_names)
combined_sheets = set(combined_xl.sheet_names)

print("\n=== COMPARISON ===")
print(f"annaB sheets: {sorted(annaB_sheets)}")
print(f"Uyi sheets: {sorted(uyi_sheets)}")
print(f"Combined sheets (done manually): {sorted(combined_sheets)}")
print(f"\nSheets in annaB but NOT in combined: {sorted(annaB_sheets - combined_sheets)}")
print(f"\nSheets in Uyi but NOT in combined: {sorted(uyi_sheets - combined_sheets)}")

# All unique sheets across both source files
all_source_sheets = sorted(annaB_sheets | uyi_sheets)
sheets_to_add = sorted((annaB_sheets | uyi_sheets) - combined_sheets)
print(f"\n=== SHEETS TO ADD ===")
print(f"Total sheets to add: {len(sheets_to_add)}")
print(f"Sheets: {sheets_to_add}")
