"""The site-specific sonic readers must emit the same turbulence fields.

Cape Cod and Rhode Island ingest EddyPro CSVs through separate readers. A
reader reading fewer columns than its sibling silently yields a site with less
data, indistinguishable from an instrument that never measured it.
"""

import numpy as np
import pandas as pd
import pytest

from windprof.anemometers import process_rhod_sonic_csv, process_sonic_c1_csv

T0 = pd.Timestamp("2024-07-15 00:00:00")
TURB_KEYS = {"ti", "tke", "std_u", "std_v", "std_w"}

COLS = ["filename", "date", "time", "wind_speed", "wind_dir", "w_unrot",
        "TKE", "u_var", "v_var", "w_var"]
UNITS = ["", "[yyyy-mm-dd]", "[HH:MM]", "[m+1s-1]", "[deg_from_north]",
         "[m+1s-1]", "[m+2s-2]", "[m+2s-2]", "[m+2s-2]", "[m+2s-2]"]


def eddypro_csv(tmp_path, ws, u_var=0.46, v_var=0.37, w_var=0.20, tke=0.52):
    """Write a minimal EddyPro-layout file: a group-header line, a names
    line, a units line, then one record whose period ends at 00:10."""
    row = ["caco.sonic.z01.01.20240715.000000.10.raw.ver_3.csv", "2024-07-15",
           "00:10", ws, 201.6, -0.08, tke, u_var, v_var, w_var]
    lines = [",".join(["file_info"] + [""] * (len(COLS) - 1)),
             ",".join(COLS), ",".join(UNITS),
             ",".join(str(v) for v in row)]
    p = tmp_path / "sonic.csv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def both(path):
    caco = process_sonic_c1_csv(path, T0, "cape_cod", instrument="z01")
    rhod = process_rhod_sonic_csv(path, T0, "rhode_island", instrument="z01")
    assert caco is not None and rhod is not None
    return caco["turbulence_profiles"], rhod["turbulence_profiles"]


def test_full_record_emits_identical_turbulence_fields(tmp_path):
    caco, rhod = both(eddypro_csv(tmp_path, ws=2.88))
    assert set(caco) == set(rhod) == {4.0}
    assert set(caco[4.0]) == set(rhod[4.0]) == TURB_KEYS
    for k in TURB_KEYS:
        assert caco[4.0][k] == pytest.approx(rhod[4.0][k])
    # Emitted values are rounded to three decimals.
    assert caco[4.0]["ti"] == pytest.approx(np.sqrt(0.46) / 2.88, abs=5e-4)
    assert caco[4.0]["std_w"] == pytest.approx(np.sqrt(0.20), abs=5e-4)


def test_missing_variances_degrade_identically(tmp_path):
    caco, rhod = both(eddypro_csv(tmp_path, ws=2.88, u_var=-9999,
                                  v_var=-9999, w_var=-9999))
    assert set(caco[4.0]) == set(rhod[4.0]) == {"tke"}


def test_physics_gate_rejects_identically(tmp_path):
    # std_u / ws = sqrt(0.46) / 0.05 > 3, so the gate rejects the record, TKE included.
    caco, rhod = both(eddypro_csv(tmp_path, ws=0.05))
    assert caco == rhod == {}
