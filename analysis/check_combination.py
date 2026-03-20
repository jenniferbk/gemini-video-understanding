import pandas as pd

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

# Let's check if you're combining both files side-by-side
sheet_name = 'Full_Jake'

print(f"=== CHECKING SHEET: {sheet_name} ===\n")

# Read from all three files
annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)

print(f"annaB shape: {annaB_df.shape}")
print(f"Uyi shape: {uyi_df.shape}")
print(f"Combined shape: {combined_df.shape}")

print(f"\nannaB columns: {list(annaB_df.columns)}")
print(f"\nUyi columns: {list(uyi_df.columns)}")
print(f"\nCombined columns: {list(combined_df.columns)}")

print("\n=== HYPOTHESIS ===")
if combined_df.shape[1] == annaB_df.shape[1] + uyi_df.shape[1]:
    print("✓ It looks like you're placing annaB and Uyi side-by-side (horizontally)")
    print(f"  annaB cols ({annaB_df.shape[1]}) + Uyi cols ({uyi_df.shape[1]}) = {annaB_df.shape[1] + uyi_df.shape[1]}")
    print(f"  Combined has {combined_df.shape[1]} columns")
elif combined_df.shape[1] > annaB_df.shape[1]:
    print(f"? Combined has {combined_df.shape[1]} columns, annaB has {annaB_df.shape[1]}, Uyi has {uyi_df.shape[1]}")
    print("  Might be some spacing or formatting columns added")
else:
    print("? Different combination strategy")

# Check a few more sheets to confirm the pattern
print("\n=== CHECKING OTHER SHEETS ===")
for sheet_name in ['Full_Ava', 'Full_Ben', 'Full_Faith']:
    try:
        annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
        uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
        combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)
        print(f"\n{sheet_name}:")
        print(f"  annaB: {annaB_df.shape[1]} cols, Uyi: {uyi_df.shape[1]} cols, Combined: {combined_df.shape[1]} cols")
        print(f"  Sum: {annaB_df.shape[1] + uyi_df.shape[1]}, Difference: {combined_df.shape[1] - (annaB_df.shape[1] + uyi_df.shape[1])}")
    except Exception as e:
        print(f"\n{sheet_name}: Error - {e}")
