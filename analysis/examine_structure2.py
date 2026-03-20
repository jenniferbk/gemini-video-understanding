import pandas as pd
import numpy as np

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

# Examine Full_Jake in detail
sheet_name = 'Full_Jake'

annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)

print(f"=== CHECKING ALL 12 COMBINED COLUMNS ===\n")

for i in range(combined_df.shape[1]):
    col_data = combined_df.iloc[:, i]
    matched = False

    # Check annaB
    for j in range(annaB_df.shape[1]):
        if col_data.equals(annaB_df.iloc[:, j]):
            print(f"Combined col {i:2d} = annaB col {j}")
            matched = True
            break

    if not matched:
        # Check Uyi
        for j in range(uyi_df.shape[1]):
            if col_data.equals(uyi_df.iloc[:, j]):
                print(f"Combined col {i:2d} = Uyi col {j}")
                matched = True
                break

    if not matched:
        if col_data.isna().all():
            print(f"Combined col {i:2d} = EMPTY/SPACER")
        else:
            # Check how many non-empty values
            non_empty = col_data.notna().sum()
            print(f"Combined col {i:2d} = UNIQUE/JENNIFER ({non_empty} non-empty cells)")

print("\n\n=== SHOWING ACTUAL VALUES FOR FIRST FEW DATA ROWS ===")
# Skip header rows and show actual data
print("\nCombined row 20-25 (all 12 columns):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)
for i in range(20, min(25, len(combined_df))):
    print(f"\nRow {i}:")
    for j in range(combined_df.shape[1]):
        val = combined_df.iloc[i, j]
        if pd.notna(val):
            val_str = str(val)[:40]
            print(f"  Col {j:2d}: {val_str}")
