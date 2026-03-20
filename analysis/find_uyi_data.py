import pandas as pd

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

sheet_name = 'Full_Jake'

annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)

print("=== LOOKING FOR UYI'S DATA IN COMBINED ===\n")

# First, let's see if Anna and Uyi have different data
print("Are annaB and Uyi files identical?")
for col_idx in range(min(annaB_df.shape[1], uyi_df.shape[1])):
    if annaB_df.iloc[:, col_idx].equals(uyi_df.iloc[:, col_idx]):
        print(f"  Column {col_idx}: IDENTICAL")
    else:
        print(f"  Column {col_idx}: DIFFERENT")
        # Show some sample differences
        for row_idx in range(min(30, len(annaB_df))):
            anna_val = annaB_df.iloc[row_idx, col_idx]
            uyi_val = uyi_df.iloc[row_idx, col_idx]
            if str(anna_val) != str(uyi_val):
                print(f"    Row {row_idx}: Anna='{str(anna_val)[:50]}' vs Uyi='{str(uyi_val)[:50]}'")
                if row_idx > 5:  # Only show first few differences
                    print("    ... (more differences)")
                    break

print("\n=== CHECKING COMBINED COLUMNS AGAINST UYI ===\n")

# Now specifically check which combined columns match which Uyi columns
for i in range(combined_df.shape[1]):
    for j in range(uyi_df.shape[1]):
        if combined_df.iloc[:, i].equals(uyi_df.iloc[:, j]):
            print(f"Combined col {i} = Uyi col {j}")

print("\n=== SAMPLE DATA FROM EACH FILE ===")
print("\nannaB col 0, row 20:")
print(annaB_df.iloc[20, 0])
print("\nUyi col 0, row 20:")
print(uyi_df.iloc[20, 0])
print("\nCombined col 0, row 20:")
print(combined_df.iloc[20, 0])
print("\nCombined col 1, row 20:")
print(combined_df.iloc[20, 1])
