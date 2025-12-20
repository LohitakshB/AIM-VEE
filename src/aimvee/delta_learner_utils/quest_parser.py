import re
import os

def QUEST_parser(dat_path):
    info = {}
    with open(dat_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            s = line.strip()
            if s.startswith("# Molecule"):
                info["molecule"] = s.split(":",1)[1].strip()
            elif s.startswith("# comment") or s.startswith("# Comment"):
                info["comment"] = s.split(":",1)[1].strip()
            elif s.startswith("# code"):
                info["code"] = s.split(":",1)[1].strip()
            elif s.startswith("# method"):
                info["method"] = s.split(":",1)[1].strip()
            elif s.startswith("# geom"):
                info["geom_descr"] = s.split(":",1)[1].strip()
            elif s.startswith("# set"):
                val = s.split(":",1)[1].strip()
                info["set"] = val
                m = re.search(r"QUEST#(\d+),?(\d+)?", val)
                if m:
                    info["set_num"] = int(m.group(1))
                    info["set_index"] = int(m.group(2)) if m.group(2) else None
    return info



def _parse_transitions(dat_path):
    """
    Parse the transitions block of a QUEST .dat file.
    Returns a list of dicts with per-transition info.
    """
    rows = []
    with open(dat_path) as f:
        in_data = False
        for line in f:
            # detect header of data table
            if line.lstrip().startswith("# Number"):
                in_data = True
                continue

            if not in_data:
                continue

            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                # skip comment / separator
                continue

            tokens = line.split()
            if len(tokens) < 7:
                continue

            try:
                ini_num = int(tokens[0])
                ini_spin = int(tokens[1])
                ini_symm = tokens[2]

                fin_num = int(tokens[3])
                fin_spin = int(tokens[4])
                fin_symm = tokens[5]
            except ValueError:
                # Something malformed; skip row
                continue

            # Find first token after index 5 that is a float → energy
            energy_idx = None
            for i in range(6, len(tokens)):
                try:
                    float(tokens[i])
                    energy_idx = i
                    break
                except ValueError:
                    continue

            if energy_idx is None:
                continue

            # transition type is everything between symm and energy
            type_tokens = tokens[6:energy_idx]
            trans_type = " ".join(type_tokens) if type_tokens else ""

            # energy
            energy_eV = float(tokens[energy_idx])

            # optional: %T1, osc strength, unsafe flag
            remaining = tokens[energy_idx + 1:]

            def _float_or_none(x):
                if x in ("_", "-", "NA", "nan"):
                    return None
                try:
                    return float(x)
                except ValueError:
                    return None

            percent_T1 = _float_or_none(remaining[0]) if len(remaining) >= 1 else None
            osc_strength = _float_or_none(remaining[1]) if len(remaining) >= 2 else None

            unsafe = None
            if len(remaining) >= 3:
                uv = remaining[2].lower()
                if uv == "true":
                    unsafe = True
                elif uv == "false":
                    unsafe = False

            rows.append(
                {
                    "initial_number": ini_num,
                    "initial_spin": ini_spin,
                    "initial_symm": ini_symm,
                    "final_number": fin_num,
                    "final_spin": fin_spin,
                    "final_symm": fin_symm,
                    "transition_type": trans_type,
                    "energy_eV": energy_eV,
                    "percent_T1": percent_T1,
                    "osc_strength": osc_strength,
                    "unsafe": unsafe,
                }
            )

    return rows


def _normalize_name(name: str) -> str:
    """Helper: normalize molecule / filename for matching."""
    return re.sub(r"[\s\-]", "", name).lower()


def find_geometry_file(quest_root, molecule, set_num=None):
    """
    Try to locate the corresponding .xyz geometry for a molecule.
    First tries QUEST{set_num}, then all QUEST* dirs.
    Returns full path or None.
    """
    structures_root = os.path.join(quest_root, "structures")
    if not os.path.isdir(structures_root):
        return None

    target_norm = _normalize_name(molecule)

    def search_in_dir(dir_path):
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith(".xyz"):
                continue
            base = os.path.splitext(fn)[0]
            if _normalize_name(base) == target_norm:
                return os.path.join(dir_path, fn)
        return None

    # 1) Try specific QUEST set first if provided
    if set_num is not None:
        quest_dir = os.path.join(structures_root, f"QUEST{set_num}")
        if os.path.isdir(quest_dir):
            p = search_in_dir(quest_dir)
            if p is not None:
                return p

    # 2) Fallback: search all QUEST* subdirs
    for sub in os.listdir(structures_root):
        if not sub.startswith("QUEST"):
            continue
        quest_dir = os.path.join(structures_root, sub)
        if not os.path.isdir(quest_dir):
            continue
        p = search_in_dir(quest_dir)
        if p is not None:
            return p

    return None


def parse_quest_file_to_rows(dat_path, quest_root):
    """
    Parse a single QUEST .dat file into a list of full rows:
    header + per-transition + geometry path.
    """
    header = QUEST_parser(dat_path)
    transitions = _parse_transitions(dat_path)

    geom_path = find_geometry_file(
        quest_root=quest_root,
        molecule=header.get("molecule", ""),
        set_num=header.get("set_num"),
    )
    geom_rel = (
        os.path.relpath(geom_path, quest_root) if geom_path is not None else None
    )

    rows = []
    for t in transitions:
        row = {
            "filename": os.path.basename(dat_path),
            "molecule": header.get("molecule"),
            "comment": header.get("comment"),
            "code": header.get("code"),
            "method": header.get("method"),
            "geom_descr": header.get("geom_descr"),
            "set": header.get("set"),
            "set_num": header.get("set_num"),
            "set_index": header.get("set_index"),
            "geom_file": geom_rel,
        }
        row.update(t)  # add transition fields
        rows.append(row)

    return rows
