import os
import numpy as np
from sklearn.utils import shuffle


def expand_geom_fid_state_block(
    X_block,
    EV_block,
    mol_ids_block,
    geom_ids_block,
    data_dir,
    molecules,
    reps_dir,
    qemfi_dir,
    n_fids,
    n_states,
):
    """
    Expand geometry-level blocks:

      X_block       : (n_geom_block, d_rep)
      EV_block      : (n_geom_block, n_fids, n_states)
      mol_ids_block : (n_geom_block,)
      geom_ids_block: (n_geom_block,)

    To per-(geom, fid, state) rows:

      X_expanded : (n_geom_block * n_fids * n_states, d_rep + 2)
                   last 2 cols = fid_id, state_id
      y_expanded : (n_geom_block * n_fids * n_states,)
      fid_ids    : (n_geom_block * n_fids * n_states,)
      state_ids  : (n_geom_block * n_fids * n_states,)
      mol_exp    : ...
      geom_exp   : ...
    """
    n_geom, d_rep = X_block.shape
    # Use provided n_fids/n_states
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
                # method rep
                X_expanded[k, :d_rep] = X_block[g]
                # set fid/state indices
                X_expanded[k, d_rep]     = f
                X_expanded[k, d_rep + 1] = s

                # target value
                y_expanded[k] = EV_block[g, f, s]

                # set IDs
                fid_ids[k]   = f
                state_ids[k] = s
                mol_exp[k]   = mol_ids_block[g]
                geom_exp[k]  = geom_ids_block[g]
                k += 1

    return X_expanded, y_expanded, fid_ids, state_ids, mol_exp, geom_exp


def prep_data(
    data_dir: str,
    molecules,
    reps_dir: str,
    qemfi_dir: str,
    method: str,
    n_geom_per_mol: int = 15000,
    
):
    """
    Prepare {method}-based EV dataset with shared geometry split.

    - Uses ALL fidelities & states for TRAIN.
    - VAL/TEST filtered to highest fidelity.
    - Saves:
        {method}_train.npy, {method}_val.npy, {method}_test.npy
        EV_train.npy, EV_val.npy, EV_test.npy

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        idx_train_rows, idx_val_rows, idx_test_rows,
        mol_ids_all, geom_ids_all,
        n_fids, n_states
    """
    os.makedirs(data_dir, exist_ok=True)

    total_geoms = len(molecules) * n_geom_per_mol

    # use last molecule reps to get max feature dim
    example_rep_path = os.path.join(reps_dir, f"QeMFi_{molecules[-1]}_{method}.npy")
    example_rep      = np.load(example_rep_path)
    max_dim          = example_rep.shape[1]

    # determine n_fids and n_states from one QeMFi npz
    sample_npz_path = os.path.join(qemfi_dir, f"QeMFi_{molecules[0]}.npz")
    sample_data     = np.load(sample_npz_path, allow_pickle=True)
    EV_sample       = sample_data["EV"]   # (15000, n_fids, n_states)

    n_fids   = EV_sample.shape[1]
    n_states = EV_sample.shape[2]

    # allocate geometry-level storage
    X_all_geom   = np.zeros((total_geoms, max_dim), dtype=float)
    mol_ids_all  = np.zeros(total_geoms, dtype=np.int64)
    geom_ids_all = np.zeros(total_geoms, dtype=np.int64)
    EV_all_geom  = np.zeros((total_geoms, n_fids, n_states), dtype=float)

    start_geom = 0

    # 1) per-molecule shuffle and stack
    for mol_id, mol in enumerate(molecules):
        end_geom = start_geom + n_geom_per_mol

        # local shuffled indices
        idx_local = np.arange(n_geom_per_mol)
        idx_local = shuffle(idx_local, random_state=42)

        # load method representations
        rep_path = os.path.join(reps_dir, f"QeMFi_{mol}_{method}.npy")
        reps_mol = np.load(rep_path)  # (15000, d_m)
        d_m      = reps_mol.shape[1]

        reps_mol_shuffled = reps_mol[idx_local, :]
        X_all_geom[start_geom:end_geom, :d_m] = reps_mol_shuffled

        # load EV block
        npz_path = os.path.join(qemfi_dir, f"QeMFi_{mol}.npz")
        data     = np.load(npz_path, allow_pickle=True)

        EV_full = data["EV"]  # (15000, n_fids, n_states)
        assert EV_full.shape[1] == n_fids and EV_full.shape[2] == n_states, \
            f"EV shape mismatch for molecule {mol}: expected (*,{n_fids},{n_states}), got {EV_full.shape}"

        EV_mol = EV_full[idx_local, :, :]
        EV_all_geom[start_geom:end_geom, :, :] = EV_mol

        mol_ids_all[start_geom:end_geom]  = mol_id
        geom_ids_all[start_geom:end_geom] = idx_local

        start_geom = end_geom

    # 2) global shuffle
    rng  = np.random.RandomState(42)
    perm = rng.permutation(total_geoms)

    X_all_geom   = X_all_geom[perm]
    EV_all_geom  = EV_all_geom[perm]
    mol_ids_all  = mol_ids_all[perm]
    geom_ids_all = geom_ids_all[perm]

    # 3) geometry split
    n_train_geoms = 120_000
    n_val_geoms   = 2_000
    n_test_geoms  = total_geoms - n_train_geoms - n_val_geoms  # 13_000

    idx_train_rows = np.arange(0, n_train_geoms)
    idx_val_rows   = np.arange(n_train_geoms, n_train_geoms + n_val_geoms)
    idx_test_rows  = np.arange(n_train_geoms + n_val_geoms, total_geoms)

    # 4) slice geometry blocks
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

    # 5) expand geom×fid×state to rows
    X_train, y_train, fid_train_ids, state_train_ids, mol_train_exp, geom_train_exp = expand_geom_fid_state_block(
        X_train_geom, EV_train_geom, mol_train_geom, geom_train_geom,
        data_dir, molecules, reps_dir, qemfi_dir, n_fids, n_states
    )
    X_val, y_val, fid_val_ids, state_val_ids, mol_val_exp, geom_val_exp = expand_geom_fid_state_block(
        X_val_geom, EV_val_geom, mol_val_geom, geom_val_geom,
        data_dir, molecules, reps_dir, qemfi_dir, n_fids, n_states
    )
    X_test, y_test, fid_test_ids, state_test_ids, mol_test_exp, geom_test_exp = expand_geom_fid_state_block(
        X_test_geom, EV_test_geom, mol_test_geom, geom_test_geom,
        data_dir, molecules, reps_dir, qemfi_dir, n_fids, n_states
    )

    # 6) Filter VAL/TEST to highest fidelity
    high_fid_idx = n_fids - 1

    val_mask  = (fid_val_ids  == high_fid_idx)
    test_mask = (fid_test_ids == high_fid_idx)

    X_val        = X_val[val_mask]
    y_val        = y_val[val_mask]
    # (fid_val_ids, state_val_ids, etc. kept if you need them later)

    X_test        = X_test[test_mask]
    y_test        = y_test[test_mask]

    # 7) Save feature matrices (method_*.npy; fid/state last cols)
    np.save(os.path.join(data_dir, f"{method}_train.npy"), X_train)
    np.save(os.path.join(data_dir, f"{method}_val.npy"),   X_val)
    np.save(os.path.join(data_dir, f"{method}_test.npy"),  X_test)

    # 8) Save EV targets
    np.save(os.path.join(data_dir, "EV_train.npy"), y_train)
    np.save(os.path.join(data_dir, "EV_val.npy"),   y_val)
    np.save(os.path.join(data_dir, "EV_test.npy"),  y_test)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        idx_train_rows, idx_val_rows, idx_test_rows,
        mol_ids_all, geom_ids_all,
        n_fids, n_states,
    )
