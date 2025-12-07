import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if ROOT not in sys.path:
    sys.path.append(ROOT)


from src.aimvee.vee_predictor_utils.qemfi_prep import prep_data_cm_ev

MOLECULES = [
    "urea", "acrolein", "alanine", "sma", "nitrophenol",
    "urocanic", "dmabn", "thymine", "o-hbdi"
]

ROOT      = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/vee_predictor"
QEMFI_DIR = os.path.join(ROOT,"QeMFi")       # QeMFi_*.npz
REPS_DIR  = os.path.join(ROOT, "QeMFi_cm")    # QeMFi_*_CM.npy
DATA_DIR  = os.path.join(ROOT, "Data")


def main():
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        idx_train_rows, idx_val_rows, idx_test_rows,
        mol_ids_all, geom_ids_all,
        n_fids, n_states,
    ) = prep_data_cm_ev(
        data_dir=DATA_DIR,
        molecules=MOLECULES,
        reps_dir=REPS_DIR,
        qemfi_dir=QEMFI_DIR,
    )

    print("CM EV data prepared.")
    print("  X_train shape:", X_train.shape)
    print("  X_val   shape:", X_val.shape)
    print("  X_test  shape:", X_test.shape)
    print("  y_train shape:", y_train.shape)
    print("  y_val   shape:", y_val.shape)
    print("  y_test  shape:", y_test.shape)
    print("  N_FIDS:", n_fids, " N_STATES:", n_states)


if __name__ == "__main__":
    main()
