import pandas as pd

INPUT_CSV = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_parsed.csv"
OUTPUT_CSV = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_parsed_CC3_aug-cc-pVTZ.csv"

def main():

    df = pd.read_csv(INPUT_CSV)

    # 1) Filter rows where method contains BOTH "CC3" and "aug-cc-pVTZ"
    mask_method = df["method"].str.contains("CC3") & df["method"].str.contains("aug-cc-pVTZ")
    df = df[mask_method].copy()

    # 2) Drop rows where unsafe == True
    # (Handles string "True", boolean True, etc.)
    mask_safe = ~(df["unsafe"].astype(str).str.lower() == "true")
    df = df[mask_safe].copy()

    # 3) Drop rows where geom_file is NaN or empty
    mask_geom = df["geom_file"].notna() & (df["geom_file"].astype(str).str.strip() != "")
    df = df[mask_geom].copy()

    # Save result
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved filtered dataset to: {OUTPUT_CSV}")
    print(f"Remaining rows: {len(df)}")
    print(f"Molecules included: {df['molecule'].nunique()}")

if __name__ == "__main__":
    main()
3