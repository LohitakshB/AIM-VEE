import os
import sys
import csv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.aimvee.delta_learner_utils.quest_parser import parse_quest_file_to_rows



# QUEST_db root (contains abs/ and structures/)
INPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_db"
ABS_DIR   = os.path.join(INPUT_DIR, "abs")
OUTPUT_CSV = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_parsed.csv"


def main():
    all_rows = []

    for fname in os.listdir(ABS_DIR):
        if not fname.endswith(".dat"):
            continue

        dat_path = os.path.join(ABS_DIR, fname)
        rows = parse_quest_file_to_rows(dat_path, quest_root=INPUT_DIR)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows parsed; check INPUT_DIR & abs/ contents.")
        return

    # Use union of keys across all rows for fieldnames
    fieldnames = sorted({k for row in all_rows for k in row.keys()})

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Parsed {len(all_rows)} transitions from {ABS_DIR}")
    print(f"CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
