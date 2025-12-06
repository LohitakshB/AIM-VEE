#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data prep for:
1) MFML baseline (CM or SLATM reps + EV/SCF energies)
2) GNN multi-fidelity *index* dataset (geometry + fidelity indices only).

For EV:
    - Uses ALL fidelities present in QeMFi
    - Each (geometry, fidelity, state) is its own row.
    - Appends fid_id and state_id as the last two feature columns in X_*.
    - BUT: validation and test sets are FILTERED to highest fidelity (TZVP).
"""

import os
import numpy as np
from sklearn.utils import shuffle

MOLECULES = [
    'urea', 'acrolein', 'alanine', 'sma', 'nitrophenol',
    'urocanic', 'dmabn', 'thymine', 'o-hbdi'
]

# These will be set from the data
N_FIDS   = None
N_STATES = None

ROOT          = "/Users/lohitakshbadarala/Desktop/AIM-VEE"
QEMFI_DIR     = os.path.join(ROOT, "data", "QeMFi")          # QeMFi_*.npz
REPS_DIR_CM   = os.path.join(ROOT, "data", "QeMFi_cm")       # QeMFi_*_CM.npy
REPS_DIR_SLATM= os.path.join(ROOT, "data", "QeMFi_slatm")    # QeMFi_*_SLATM.npy
DATA_DIR      = os.path.join(ROOT, "data", "Data")
OUT_GNN_INDEX = os.path.join(ROOT, "data", "gnn_index_multifid.npz")


def _expand_geom_fid_state_block(X_block, EV_block, mol_ids_block, geom_ids_block):
    """
    Given geometry-level block:

      X_block       : (n_geom_block, d_rep)
      EV_block      : (n_geom_block, n_fids, n_states)
      mol_ids_block : (n_geom_block,)
      geom_ids_block: (n_geom_block,)

    Produce per-(geom, fid, state) rows:

      X_expanded : (n_geom_block * n_fids * n_states, d_rep + 2)
                   - last two columns = fid_id, state_id
      y_expanded : (n_geom_block * n_fids * n_states,)
      fid_ids    : (n_geom_block * n_fids * n_states,)
      state_ids  : (n_geom_block * n_fids * n_states,)
      mol_exp    : (n_geom_block * n_fids * n_states,)
      geom_exp   : (n_geom_block * n_fids * n_states,)
    """
    n_geom, d_rep = X_block.shape
    _, n_fids, n_states = EV_block.shape

    n_samples = n_geom * n_fids * n_states

    X_expanded = np.zeros((n_samples, d_rep + 2), dtype=float)
    y_expanded = np.zeros(n_samples, dtype=float)
    fid_ids    = np.zeros(n_samples, dtype=np.int32)
    state_ids  = np.zeros(n_samples, dtype=np.int32)
    mol_exp    = np.zeros(n_samples, dtype=np.int64)
    geom_exp   = np.zeros(n_samples, dtype=np.int64)

    k = 0
    for g in range(n_geom):
        for f in range(n_fids):
            for s in range(n_states):
                # copy representation
                X_expanded[k, :d_rep] = X_block[g]
                # append fid_id and state_id as features
                X_expanded[k, d_rep]     = f
                X_expanded[k, d_rep + 1] = s

                # target
                y_expanded[k] = EV_block[g, f, s]

                # IDs
                fid_ids[k]   = f
                state_ids[k] = s
                mol_exp[k]   = mol_ids_block[g]
                geom_exp[k]  = geom_ids_block[g]
                k += 1

    return X_expanded, y_expanded, fid_ids, state_ids, mol_exp, geom_exp


def prep_data_mfml(prop: str = 'EV', rep: str = 'CM'):
    """
    Prepar data with a shared geometry split, and
    record mapping from rows -> (mol_id, geom_id).

    For prop == 'EV':
        - Uses ALL fidelities & ALL states present in QeMFi.
        - Each (geometry, fidelity, state) becomes its own row.
        - fid_id and state_id are appended as last two features in X_*.
        - TRAIN uses all fidelities; VAL/TEST are filtered to highest fidelity.

    For prop == 'SCF':
        - 1 row per geometry, multi-fid y_all
    """
    global N_FIDS, N_STATES

    os.makedirs(DATA_DIR, exist_ok=True)

    # choose the correct reps directory
    if rep.upper() == 'CM':
        reps_dir = REPS_DIR_CM
    elif rep.upper() == 'SLATM':
        reps_dir = REPS_DIR_SLATM
    else:
        raise ValueError(f"Unknown rep '{rep}'. Use 'CM' or 'SLATM'.")

    # 15000 geometries per molecule, 9 molecules
    n_geom_per_mol = 15000
    total_geoms    = len(MOLECULES) * n_geom_per_mol  # 135000

    # load one rep file (o-hbdi) to get max feature dim
    example_rep_path = os.path.join(reps_dir, f"QeMFi_{MOLECULES[-1]}_{rep}.npy")
    example_rep      = np.load(example_rep_path)
    max_dim          = example_rep.shape[1]

    # determine N_FIDS and N_STATES from the data
    sample_npz_path = os.path.join(QEMFI_DIR, f"QeMFi_{MOLECULES[0]}.npz")
    sample_data     = np.load(sample_npz_path, allow_pickle=True)
    if prop.upper() == 'EV':
        EV_sample       = sample_data['EV']            # shape (15000, n_fids, n_states)
        n_fids_full     = EV_sample.shape[1]
        n_states_full   = EV_sample.shape[2]
        N_FIDS   = n_fids_full
        N_STATES = n_states_full
    else:
        SCF_sample  = sample_data['SCF']               # shape (15000, n_fids)
        n_fids_full = SCF_sample.shape[1]
        N_FIDS      = n_fids_full

    
    X_all_geom   = np.zeros((total_geoms, max_dim), dtype=float)
    mol_ids_all  = np.zeros(total_geoms, dtype=np.int64)
    geom_ids_all = np.zeros(total_geoms, dtype=np.int64)

    if prop.upper() == 'EV':
        EV_all_geom = np.zeros((total_geoms, N_FIDS, N_STATES), dtype=float)
    elif prop.upper() == 'SCF':
        y_all_geom  = np.zeros((total_geoms, N_FIDS), dtype=float)

    start_geom = 0

    # 1) per-molecule internal shuffle and stacking (geometry-level)
    for mol_id, mol in enumerate(MOLECULES):
        end_geom = start_geom + n_geom_per_mol

        # per-molecule geometry indices
        idx_local = np.arange(n_geom_per_mol)
        idx_local = shuffle(idx_local, random_state=42)

        # load representations
        rep_path = os.path.join(reps_dir, f"QeMFi_{mol}_{rep}.npy")
        reps_mol = np.load(rep_path)  # shape (15000, d_m)
        d_m      = reps_mol.shape[1]

        reps_mol_shuffled = reps_mol[idx_local, :]

        X_all_geom[start_geom:end_geom, :d_m] = reps_mol_shuffled

        # load EV or SCF from QeMFi
        npz_path = os.path.join(QEMFI_DIR, f"QeMFi_{mol}.npz")
        data     = np.load(npz_path, allow_pickle=True)

        if prop.upper() == 'EV':
            EV_full = data['EV']  # (15000, N_FIDS, N_STATES)
            assert EV_full.shape[1] == N_FIDS and EV_full.shape[2] == N_STATES, \
                f"EV shape mismatch for molecule {mol}: expected (*,{N_FIDS},{N_STATES}), got {EV_full.shape}"
            EV_mol = EV_full[idx_local, :, :]
            EV_all_geom[start_geom:end_geom, :, :] = EV_mol

        elif prop.upper() == 'SCF':
            SCF = data['SCF'][:, :N_FIDS]
            SCF = SCF[idx_local, :]
            y_all_geom[start_geom:end_geom, :] = SCF

        else:
            raise ValueError(f"Unknown prop '{prop}', use 'EV' or 'SCF'.")

        mol_ids_all[start_geom:end_geom]  = mol_id
        geom_ids_all[start_geom:end_geom] = idx_local

        start_geom = end_geom

    # 2) global shuffle across all geometries, once, with fixed seed
    rng  = np.random.RandomState(42)
    perm = rng.permutation(total_geoms)

    X_all_geom   = X_all_geom[perm]
    mol_ids_all  = mol_ids_all[perm]
    geom_ids_all = geom_ids_all[perm]

    if prop.upper() == 'EV':
        EV_all_geom = EV_all_geom[perm]
    elif prop.upper() == 'SCF':
        y_all_geom  = y_all_geom[perm]

    # 3) split on *geometries*
    n_train_geoms = 120_000
    n_val_geoms   = 2_000
    n_test_geoms  = total_geoms - n_train_geoms - n_val_geoms  # 13_000

    idx_train_rows = np.arange(0, n_train_geoms)
    idx_val_rows   = np.arange(n_train_geoms, n_train_geoms + n_val_geoms)
    idx_test_rows  = np.arange(n_train_geoms + n_val_geoms, total_geoms)

    # 4) expand to per-(geom, fid, state) rows for EV
    if prop.upper() == 'EV':
        X_train_geom = X_all_geom[idx_train_rows]
        X_val_geom   = X_all_geom[idx_val_rows]
        X_test_geom  = X_all_geom[idx_test_rows]

        EV_train_geom = EV_all_geom[idx_train_rows]
        EV_val_geom   = EV_all_geom[idx_val_rows]
        EV_test_geom  = EV_all_geom[idx_test_rows]

        mol_train_geom  = mol_ids_all[idx_train_rows]
        mol_val_geom    = mol_ids_all[idx_val_rows]
        mol_test_geom   = mol_ids_all[idx_test_rows]

        geom_train_geom = geom_ids_all[idx_train_rows]
        geom_val_geom   = geom_ids_all[idx_val_rows]
        geom_test_geom  = geom_ids_all[idx_test_rows]

        # expand each block: geom×fid×state → rows
        X_train, y_train, fid_train_ids, state_train_ids, mol_train_exp, geom_train_exp = _expand_geom_fid_state_block(
            X_train_geom, EV_train_geom, mol_train_geom, geom_train_geom
        )
        X_val, y_val, fid_val_ids, state_val_ids, mol_val_exp, geom_val_exp = _expand_geom_fid_state_block(
            X_val_geom, EV_val_geom, mol_val_geom, geom_val_geom
        )
        X_test, y_test, fid_test_ids, state_test_ids, mol_test_exp, geom_test_exp = _expand_geom_fid_state_block(
            X_test_geom, EV_test_geom, mol_test_geom, geom_test_geom
        )

        # Filter VAL and TEST to highest fidelity (TZVP)
        high_fid_idx = N_FIDS - 1 

        val_mask  = (fid_val_ids  == high_fid_idx)
        test_mask = (fid_test_ids == high_fid_idx)

        X_val        = X_val[val_mask]
        y_val        = y_val[val_mask]
        fid_val_ids  = fid_val_ids[val_mask]
        state_val_ids= state_val_ids[val_mask]
        mol_val_exp  = mol_val_exp[val_mask]
        geom_val_exp = geom_val_exp[val_mask]

        X_test        = X_test[test_mask]
        y_test        = y_test[test_mask]
        fid_test_ids  = fid_test_ids[test_mask]
        state_test_ids= state_test_ids[test_mask]
        mol_test_exp  = mol_test_exp[test_mask]
        geom_test_exp = geom_test_exp[test_mask]

        # Save feature matrices (with fid_id, state_id as last two columns)
        np.save(os.path.join(DATA_DIR, f"{rep}_train.npy"), X_train)
        np.save(os.path.join(DATA_DIR, f"{rep}_val.npy"),   X_val)
        np.save(os.path.join(DATA_DIR, f"{rep}_test.npy"),  X_test)

        # Save targets (scalar EV)
        np.save(os.path.join(DATA_DIR, f"{prop}_train.npy"), y_train)
        np.save(os.path.join(DATA_DIR, f"{prop}_val.npy"),   y_val)
        np.save(os.path.join(DATA_DIR, f"{prop}_test.npy"),  y_test)

        # Save fid/state ids separately (handy for debugging/analysis)
        np.save(os.path.join(DATA_DIR, "fid_train_ids.npy"),   fid_train_ids)
        np.save(os.path.join(DATA_DIR, "fid_val_ids.npy"),     fid_val_ids)
        np.save(os.path.join(DATA_DIR, "fid_test_ids.npy"),    fid_test_ids)
        np.save(os.path.join(DATA_DIR, "state_train_ids.npy"), state_train_ids)
        np.save(os.path.join(DATA_DIR, "state_val_ids.npy"),   state_val_ids)
        np.save(os.path.join(DATA_DIR, "state_test_ids.npy"),  state_test_ids)

        # Save geometry-level mapping and splits (still geometry-based)
        np.save(os.path.join(DATA_DIR, "idx_train_rows.npy"), idx_train_rows)
        np.save(os.path.join(DATA_DIR, "idx_val_rows.npy"),   idx_val_rows)
        np.save(os.path.join(DATA_DIR, "idx_test_rows.npy"),  idx_test_rows)
        np.save(os.path.join(DATA_DIR, "mol_ids_all.npy"),    mol_ids_all)
        np.save(os.path.join(DATA_DIR, "geom_ids_all.npy"),   geom_ids_all)

        return (X_train, X_val, X_test,
                y_train, y_val, y_test,
                idx_train_rows, idx_val_rows, idx_test_rows,
                mol_ids_all, geom_ids_all)

    # ---- For SCF: keep original behavior ----
    elif prop.upper() == 'SCF':
        X_train = X_all_geom[idx_train_rows]
        X_val   = X_all_geom[idx_val_rows]
        X_test  = X_all_geom[idx_test_rows]

        y_train = np.zeros((N_FIDS,), dtype=object)
        for f in range(N_FIDS):
            y_train[f] = y_all_geom[idx_train_rows, f]

        # validation / test target = highest fidelity (last index)
        y_val  = y_all_geom[idx_val_rows,  -1]
        y_test = y_all_geom[idx_test_rows, -1]

        # save data
        np.save(os.path.join(DATA_DIR, f"{rep}_train.npy"), X_train)
        np.save(os.path.join(DATA_DIR, f"{rep}_val.npy"),   X_val)
        np.save(os.path.join(DATA_DIR, f"{rep}_test.npy"),  X_test)

        np.save(os.path.join(DATA_DIR, f"{prop}_train.npy"), y_train)
        np.save(os.path.join(DATA_DIR, f"{prop}_val.npy"),   y_val)
        np.save(os.path.join(DATA_DIR, f"{prop}_test.npy"),  y_test)

        np.save(os.path.join(DATA_DIR, "idx_train_rows.npy"), idx_train_rows)
        np.save(os.path.join(DATA_DIR, "idx_val_rows.npy"),   idx_val_rows)
        np.save(os.path.join(DATA_DIR, "idx_test_rows.npy"),  idx_test_rows)
        np.save(os.path.join(DATA_DIR, "mol_ids_all.npy"),    mol_ids_all)
        np.save(os.path.join(DATA_DIR, "geom_ids_all.npy"),   geom_ids_all)

        return (X_train, X_val, X_test,
                y_train, y_val, y_test,
                idx_train_rows, idx_val_rows, idx_test_rows,
                mol_ids_all, geom_ids_all)


def prep_data_gnn_indices(idx_train_rows,
                          idx_val_rows,
                          idx_test_rows,
                          mol_ids_all,
                          geom_ids_all):
    """
    GNN *index* dataset, geometry-level only (unchanged).
    """

    mol_ids_all  = np.asarray(mol_ids_all,  dtype=np.int32)
    geom_ids_all = np.asarray(geom_ids_all, dtype=np.int32)

    train_rows = np.asarray(idx_train_rows, dtype=np.int32)
    val_rows   = np.asarray(idx_val_rows,   dtype=np.int32)
    test_rows  = np.asarray(idx_test_rows,  dtype=np.int32)

    n_train_geoms = len(train_rows)
    n_val_geoms   = len(val_rows)
    n_test_geoms  = len(test_rows)

    n_train_samples = n_train_geoms * N_FIDS
    n_val_samples   = n_val_geoms   * N_FIDS
    n_test_samples  = n_test_geoms  * N_FIDS

    train_geom_idx = np.empty(n_train_samples, dtype=np.int32)
    train_fid      = np.empty(n_train_samples, dtype=np.int8)

    val_geom_idx   = np.empty(n_val_samples,   dtype=np.int32)
    val_fid        = np.empty(n_val_samples,   dtype=np.int8)

    test_geom_idx  = np.empty(n_test_samples,  dtype=np.int32)
    test_fid       = np.empty(n_test_samples,  dtype=np.int8)

    # fill training index arrays
    k = 0
    for r in train_rows:
        for f in range(N_FIDS):
            train_geom_idx[k] = r
            train_fid[k]      = f
            k += 1

    # fill validation index arrays
    k = 0
    for r in val_rows:
        for f in range(N_FIDS):
            val_geom_idx[k] = r
            val_fid[k]      = f
            k += 1

    # fill test index arrays
    k = 0
    for r in test_rows:
        for f in range(N_FIDS):
            test_geom_idx[k] = r
            test_fid[k]      = f
            k += 1

    np.savez_compressed(
        OUT_GNN_INDEX,
        train_geom_idx=train_geom_idx,
        train_fid=train_fid,
        val_geom_idx=val_geom_idx,
        val_fid=val_fid,
        test_geom_idx=test_geom_idx,
        test_fid=test_fid,
        mol_ids_all=mol_ids_all,
        geom_ids_all=geom_ids_all,
        N_FIDS=N_FIDS,
        N_STATES=N_STATES,
        MOLECULES=np.array(MOLECULES, dtype=object),
        QEMFI_DIR=QEMFI_DIR
    )

    return (train_geom_idx, train_fid,
            val_geom_idx,   val_fid,
            test_geom_idx,  test_fid)


if __name__ == "__main__":

    (X_train, X_val, X_test,
     y_train_mfml, y_val_mfml, y_test_mfml,
     idx_train_rows, idx_val_rows, idx_test_rows,
     mol_ids_all, geom_ids_all) = prep_data_mfml(prop='EV', rep='CM')

    print("MFML EV data prepared.")
    print("  X_train shape:", X_train.shape)
    print("  X_val   shape:", X_val.shape)
    print("  X_test  shape:", X_test.shape)
    print("  y_train_mfml shape:", y_train_mfml.shape)
    print("  y_val_mfml shape:",   y_val_mfml.shape)
    print("  y_test_mfml shape:",  y_test_mfml.shape)
    print("  N_FIDS:", N_FIDS, " N_STATES:", N_STATES)

    (train_geom_idx, train_fid,
     val_geom_idx,   val_fid,
     test_geom_idx,  test_fid) = prep_data_gnn_indices(
        idx_train_rows, idx_val_rows, idx_test_rows,
        mol_ids_all, geom_ids_all
    )

    print("GNN index dataset prepared.")
    print("  #train geometries:", len(idx_train_rows),
          ", samples (× N_FIDS):", len(train_geom_idx))
    print("  #val geometries:",   len(idx_val_rows),
          ", samples (× N_FIDS):", len(val_geom_idx))
    print("  #test geometries:",  len(idx_test_rows),
          ", samples (× N_FIDS):", len(test_geom_idx))
    print("Saved GNN index dataset to:", OUT_GNN_INDEX)
