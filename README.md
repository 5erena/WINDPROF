# WINDPROF

Multi-instrument merged wind and turbulence profile product for the WFIP3 offshore campaign (Block Island, Rhode Island, Nantucket, Cape Cod).

WINDPROF ingests Doppler lidar (scanning and profiling), radar wind profiler, and sonic anemometer data; applies instrument-specific quality control; performs VAD wind retrievals; and merges the retrievals into a single height-resolved wind and turbulence profile on a common time/height grid.

> **Just looking for the data?** The processed WFIP3 merged profiles are archived on the [DOE Wind Data Hub](https://a2e.energy.gov/) — you do not need this repository. This codebase is for users who want to understand the processing methodology or adapt the pipeline for a similar multi-instrument campaign.

## Scientific reference

Lipari, S., et al. (2026). *Integrated Wind and Turbulence Profiles for Characterization of the Atmospheric Boundary Layer.* Atmospheric Measurement Techniques (in preparation).

Methods inline in the code cite Rosenbusch (2021), Stull (1988), Kaimal & Finnigan (1994), and IEC 61400-1 (2019) where relevant.

## Supported sites and instruments

The four WFIP3 sites and their deployed instruments are listed below. *Site-specific instrument labels (z01, z02, z03) follow the WFIP3 naming convention and do not correspond to a fixed instrument role across sites — for example, the WindCube profiling lidar is labeled z03 at Nantucket, Block Island, and Narragansett but z01 at Cape Cod.*

| Site | Scanning lidar(s) | Profiling lidar | Radar | Sonic |
|------|------------------|-----------------|-------|-------|
| Nantucket (NANT)     | Halo XR+ (z01), Halo XR (z02) | Leosphere WindCube V2.1 (z03) | NOAA PSL 915 MHz | 1 height (z02, 5 m) |
| Block Island (BLOC)  | Halo XR (z01)                 | Leosphere WindCube V1 (z03)   | NOAA PSL 915 MHz | 1 height (z01, 10 m) |
| Cape Cod (CACO)      | Halo XR+ (z02)                | Leosphere WindCube V2-96 (z01) | —    | 2 heights (z01 4 m, z02 10 m) |
| Narragansett (RHOD)  | Halo Galion (z01)             | ZephIR-300 (z03)              | —    | 1 height (z01, 4 m) |

See the AMT manuscript and `windprof/config.py` for full per-instrument specifications including range, scan cycle, QC thresholds, and applied corrections.

## Installation

```bash
conda env create -f environment.yml
conda activate windprof
pip install -e .
```

## Configuration

WINDPROF reads raw instrument files from two roots and writes merged NetCDF output to a third:

```bash
export WINDPROF_DATA_PATH=/path/to/wfip3/raw          # locally curated inputs (externally sourced or pre-merged daily files: scanning lidars, radar)
export WINDPROF_ARCHIVE_PATH=/path/to/wfip3/archive   # campaign archive tree (canonical daily files as delivered: profiling lidars, sonics, surface met)
export WINDPROF_RESULTS_PATH=/path/to/wfip3/output    # merged NetCDF output
```

If unset, the three default to `./data`, `/data`, and `./results` respectively (the `/data` default reflects the WFIP3 server layout).

Site-specific instrument layouts, ground elevations, azimuth corrections, vertical-velocity sign conventions, and QC thresholds live in [`windprof/config.py`](windprof/config.py). All site-specific knowledge is intended to be config-driven; processing modules read from `config.py` rather than embedding site assumptions.

## Usage

Process a single date for a site:

```python
from windprof.pipeline import process_wind_profiles_for_date

process_wind_profiles_for_date(
    location='nantucket',
    date='2024-03-01',
)
```

Parallel processing across dates:

```bash
python -m windprof.process_parallel nantucket --start-date 2024-03-01 --end-date 2024-03-07
```

## Module layout

All processing lives in `windprof/`:

| File | Role |
| --- | --- |
| `config.py` | Per-site instrument layouts, coordinates, elevations, corrections, paths |
| `wind_analysis.py` | VAD fitting, hybrid wind speed (Rosenbusch 2021), error propagation |
| `quality_control.py` | Per-instrument QC, inter-instrument agreement flags |
| `lidar_parsers.py` | Format-specific readers for the various lidar file structures |
| `lidars.py` | Scanning and profiling lidar processing entry points |
| `radars.py` | Radar wind profiler processing (NOAA PSL 915 MHz and NetCDF formats) |
| `anemometers.py` | Sonic anemometer and surface met processing |
| `merging.py` | Hierarchical cross-instrument averaging and quality flag synthesis |
| `discovery_and_export.py` | File discovery and CF-compliant NetCDF export |
| `pipeline.py` | Top-level orchestration |
| `process_parallel.py` | Parallel merged data processing |
| `plotting.py` | Diagnostic plots and Hovmöller summaries |

## Output

Merged products are written as CF-compliant NetCDF files with per-height quality flags (`qc_wind_speed_merged`, `qc_wind_direction_merged`, `qc_vertical_velocity_merged`, `qc_turbulence_intensity_merged`) on a 0=good / 1=suspect / 2=bad / 3=no_data convention. The −9999 sentinel marks missing data.

## Data quality and corrections

All WINDPROF profiles are reported in true-north meteorological convention. Site- and instrument-specific azimuth and vertical-velocity-sign corrections are applied in `config.py`; their derivation is documented in the manuscript Methods section and in code comments. Users requiring research-grade wind direction at affected sites should consult the manuscript and the data-quality note distributed with the archived dataset on the [DOE Wind Data Hub](https://a2e.energy.gov/).

## Adapting WINDPROF for your own campaign

The pipeline is structured so that adapting it for a different campaign with different sites and instruments concentrates the work in a small number of files. In rough order of effort:

1. **Add your site to `config.py`.** Copy one of the existing `LOCATION_CONFIG` blocks and edit:
   - `coordinates` and `elevations` for each instrument
   - `wind_corrections` (azimuth offsets relative to true north, in degrees)
   - `w_sign_corrections` (vertical velocity sign multipliers, ±1)
   - `anemometer_corrections` and `anemometer_heights` if you have sonics
2. **Add per-instrument QC thresholds and scan-geometry settings.** Scanning-lidar reset thresholds, QC metrics (SNR vs. intensity), and beam-count minimums live in `lidars.py` and `quality_control.py`. Each is named-constant or comment-documented.
3. **Add file-format readers if your instrument files differ.** `lidar_parsers.py` contains format-specific readers (e.g., `_process_z03_rtd`, `_process_z03_sta`, `_process_z03_csv` for the three WindCube/ZephIR file conventions encountered across WFIP3 sites). If your instrument writes a different format, add a new reader function and dispatch to it from the appropriate processing entry point.
4. **Update `discovery_and_export.py`.** File-name patterns and per-site instrument inventories are listed here; extend them for your campaign.

## Tests

```bash
pytest
```

By default this runs lightweight import-only smoke tests plus a synthetic
VAD-recovery check. To enable the end-to-end integration test, populate
`tests/data/` with one day of public WFIP3 input files from the DOE Wind
Data Hub — see [`tests/data/README.md`](tests/data/README.md) for the
specific files and layout. The integration test then runs the full
pipeline on that day and validates the output NetCDF structurally.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
