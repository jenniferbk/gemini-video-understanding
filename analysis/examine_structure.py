import pandas as pd
import numpy as np

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

# Examine Full_Jake in detail to understand the structure
sheet_name = 'Full_Jake'

print(f"=== EXAMINING {sheet_name} IN DETAIL ===\n")

annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)

print("COMBINED FILE STRUCTURE:")
print(f"Total columns: {combined_df.shape[1]}")
print(f"\nColumn names:")
for i, col in enumerate(combined_df.columns):
    print(f"  Col {i}: {col}")

# Show first 15 rows with all columns
print("\n=== FIRST 15 ROWS OF COMBINED (showing all columns) ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
print(combined_df.head(15))

print("\n\n=== ANNA B SOURCE (first 15 rows, first 3 cols) ===")
print(annaB_df.iloc[:15, :3])

print("\n\n=== UYI SOURCE (first 15 rows, first 3 cols) ===")
print(uyi_df.iloc[:15, :3])

# Let's check if certain columns match between source and combined
print("\n\n=== MATCHING ANALYSIS ===")
print(f"annaB has {annaB_df.shape[1]} columns")
print(f"Uyi has {uyi_df.shape[1]} columns")
print(f"Combined has {combined_df.shape[1]} columns")

# Try to identify which columns come from where by comparing content
print("\n\nChecking if combined columns match source columns...")
for i in range(min(5, combined_df.shape[1])):
    col_data = combined_df.iloc[:, i]
    # Check if this matches annaB
    matches_anna = False
    matches_uyi = False

    for j in range(annaB_df.shape[1]):
        if col_data.equals(annaB_df.iloc[:, j]):
            print(f"Combined col {i} matches annaB col {j}")
            matches_anna = True
            break

    if not matches_anna:
        for j in range(uyi_df.shape[1]):
            if col_data.equals(uyi_df.iloc[:, j]):
                print(f"Combined col {i} matches Uyi col {j}")
                matches_uyi = True
                break

    if not matches_anna and not matches_uyi:
        # Check if it's all NaN (empty spacer column)
        if col_data.isna().all():
            print(f"Combined col {i} is empty (spacer)")
        else:
            print(f"Combined col {i} is UNIQUE (Jennifer's observations?)")
