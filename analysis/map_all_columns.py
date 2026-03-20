import pandas as pd

# File paths
annaB_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_annaB.xlsx'
uyi_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_Uyi.xlsx'
combined_path = '/Users/jenniferkleiman/Documents/COMS/analysis/transcripts_combined.xlsx'

sheet_name = 'Full_Jake'

annaB_df = pd.read_excel(annaB_path, sheet_name=sheet_name)
uyi_df = pd.read_excel(uyi_path, sheet_name=sheet_name)
combined_df = pd.read_excel(combined_path, sheet_name=sheet_name)

print("=== COMPLETE COLUMN MAPPING ===\n")

# Check every combined column against every source column
mapping = {}

for i in range(combined_df.shape[1]):
    mapping[i] = {"source": "JENNIFER/UNKNOWN", "anna_match": None, "uyi_match": None}

    # Check against all annaB columns
    for j in range(annaB_df.shape[1]):
        if combined_df.iloc[:, i].equals(annaB_df.iloc[:, j]):
            mapping[i]["anna_match"] = j
            mapping[i]["source"] = f"annaB col {j}"
            break

    # Check against all Uyi columns
    for j in range(uyi_df.shape[1]):
        if combined_df.iloc[:, i].equals(uyi_df.iloc[:, j]):
            mapping[i]["uyi_match"] = j
            if mapping[i]["anna_match"] is not None:
                mapping[i]["source"] = f"BOTH (annaB col {mapping[i]['anna_match']} = Uyi col {j})"
            else:
                mapping[i]["source"] = f"Uyi col {j}"
            break

    # Check if empty
    if combined_df.iloc[:, i].isna().all():
        mapping[i]["source"] = "EMPTY/SPACER"

    # Count non-empty cells for Jennifer's columns
    non_empty = combined_df.iloc[:, i].notna().sum()

    print(f"Combined col {i:2d}: {mapping[i]['source']:40s} ({non_empty} non-empty cells)")

print("\n\n=== SUMMARY ===")
print(f"annaB has {annaB_df.shape[1]} columns")
print(f"Uyi has {uyi_df.shape[1]} columns")
print(f"Combined has {combined_df.shape[1]} columns")

print("\n=== LIKELY STRUCTURE ===")
for i in range(combined_df.shape[1]):
    source = mapping[i]['source']
    if 'BOTH' in source:
        print(f"Col {i:2d}: Shared transcript data")
    elif 'annaB' in source:
        print(f"Col {i:2d}: Anna's observations/data")
    elif 'Uyi' in source:
        print(f"Col {i:2d}: Uyi's observations/data")
    elif 'EMPTY' in source:
        print(f"Col {i:2d}: Spacer column")
    else:
        print(f"Col {i:2d}: Jennifer's observations")
