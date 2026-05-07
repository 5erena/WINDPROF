# Sample data for integration testing

This directory is empty by default. To enable the integration smoke test in
`tests/test_smoke.py`, populate it with one day of public WFIP3 input files
from the DOE Wind Data Hub.

The reference test date is **2024-09-15 at Block Island** — chosen because
BLOC has the smallest instrument inventory (one scanning lidar, one
profiling lidar, one radar, surface met) and was verified to produce a
clean merged output during refactor validation.

## Layout the test expects

The pipeline's discovery code splits inputs across two roots:

```
tests/data/
├── data/                                    # WINDPROF_DATA_PATH (high-volume)
│   └── 202409/
│       ├── bloc.lidar.z01.a0.20240915.*.nc
│       └── bloc.radar.20240915.txt
└── archive/                                 # WINDPROF_ARCHIVE_PATH (low-volume)
    ├── bloc.lidar.z03.00/
    │   └── 2024/09/15/
    │       └── bloc.lidar.z03.00.20240915.*.sta
    └── bloc.met.z01.c1/
        └── 2024/09/15/
            └── bloc.met.z01.c1.20240915.*.nc
```

## How to populate

Each WFIP3 datastream is published on WDH at
`https://wdh.energy.gov/ds/wfip3/{datastream}`. Download the four files for
2024-09-15 from:

- `bloc.lidar.z01.a0` → put in `tests/data/data/202409/`
- `bloc.lidar.z03.00` → put in `tests/data/archive/bloc.lidar.z03.00/2024/09/15/`
- `bloc.met.z01.c1` → put in `tests/data/archive/bloc.met.z01.c1/2024/09/15/`
- The Block Island radar file (NOAA PSL native text format, distributed
  alongside the campaign data) → `tests/data/data/202409/bloc.radar.20240915.txt`

Total download size: roughly 5–10 MB.

## Running the integration test

Once populated:

```bash
pytest tests/test_smoke.py -v
```

The integration test (`test_integration_bloc_one_day`) runs the full
pipeline on the sample day and verifies the output NetCDF is structurally
valid: expected variables present, sensible value ranges, height grid
monotone. It does **not** require byte equivalence against a reference
file, so the test stays robust under harmless code reorganization.

If `tests/data/data/` is empty, the integration test is automatically
skipped and only the import-only smoke tests run.

## Why this isn't shipped with the repo

The campaign data is public and authoritative on WDH. Duplicating files
here would create a second source of truth that could drift from the
archive over time. Pointing every contributor at the canonical URL keeps
the test coverage anchored to the real product.
