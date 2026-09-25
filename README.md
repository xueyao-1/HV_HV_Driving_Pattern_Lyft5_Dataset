# HV-HV Driving Pattern Dataset

A labelled driving-pattern dataset derived from HV-HV (human-driver-following-human-driver)
car-following data (`trainHH_data.csv`, from the Lyft Level 5 dataset), released for general use
in driving-behaviour research. It covers all **16,829 vehicles** in the source dataset, each
broken into a sequence of labelled driving-pattern segments using a rule-based classifier over
smoothed velocity, acceleration, time headway, and relative speed.

## What's in here

| File | Rows | Size | Grain |
|---|---|---|---|
| `trajectories_labeled_all_vehicles.csv` | 3,227,190 | 268 MB | one row per (vehicle, frame) — **recommended starting point** |
| `trajectories_all_vehicles.csv` | 3,227,190 | 157 MB | one row per (vehicle, frame), no label columns |
| `segments_all_vehicles.csv` | 67,037 | 2 MB | one row per driving-pattern segment |
| `labeled_data_all_vehicles.csv` | 16,829 | 1.8 MB | one row per vehicle (the raw rule-based output) |

`trajectories_labeled_all_vehicles.csv` is a merge of the other three: every single frame already
carries the pattern it belongs to and how long that pattern lasted, so you can filter to one
vehicle and have everything in one table with no joins. The other three files are the same data
kept apart (smaller total footprint, easier to update one piece without rewriting a 268MB file);
use those if you already have a workflow for joining tables.

**Getting the two large trajectory files**: because they're over GitHub's 100MB
per-file limit, `trajectories_labeled_all_vehicles.csv` and
`trajectories_all_vehicles.csv` are not stored in this git repository directly.
Download them from this repo's **Releases** page instead (see the release
attached to this repo's latest tag); everything else here is small enough to
be tracked normally.

### `trajectories_labeled_all_vehicles.csv` — the all-in-one file

| Column | Meaning |
|---|---|
| `VehicleID` | Vehicle identifier (matches `Vehicle_ID` in the source `trainHH_data.csv`) |
| `FrameID` | 0-based frame index, local to this vehicle (0, 1, 2, ...) |
| `timestamp` | Elapsed time in seconds since this vehicle's first frame (~0.1s per frame, i.e. 10Hz) |
| `smooth_v_follow` | Smoothed following-vehicle speed (m/s) |
| `a_follow` | Following-vehicle acceleration (m/s²) |
| `time_headway` | Time headway to the lead vehicle (s) |
| `smooth_delta_v` | Smoothed relative speed, lead minus follow (m/s) |
| `segment_index` | Which driving-pattern segment this frame belongs to (0-based, per vehicle) |
| `pattern_label` | Name of the driving pattern for this frame (see the 6 patterns below) |
| `pattern_id` | Integer encoding of `pattern_label` (0-5, mapping below) |
| `pattern_start_frame` | `FrameID` where this pattern segment starts |
| `pattern_end_frame` | `FrameID` where this pattern segment ends |
| `pattern_duration` | Length of this pattern segment, in frames (≈ ×0.1 for seconds) |

### `trajectories_all_vehicles.csv` — trajectories only (no label columns)
Same as above minus the `pattern_*` columns; join to `segments_all_vehicles.csv` on
(`VehicleID`, `segment_index`) to attach labels yourself.

### `segments_all_vehicles.csv` — one row per pattern segment
`VehicleID, segment_index, pattern_label, pattern_id, start_frame, end_frame, duration`

### `labeled_data_all_vehicles.csv` — the raw rule-based output
`VehicleID, Label, Point` — `Label` is the full list of pattern names found for that vehicle
(as Python-literal strings, e.g. `['Speed_up', 'Fall_behind', 'Follow_behind']`) and `Point` the
frame indices where the algorithm detected each transition. This is upstream of the other three
files (kept for anyone who wants to re-derive segments/durations differently); the first entry in
`Label` has no preceding segment and is dropped when building the other files (i.e. segment `i`
spans `[Point[i], Point[i+1])` and takes its label from `Label[i+1]`).

## The 6 driving patterns

| `pattern_id` | `pattern_label` | Meaning |
|---|---|---|
| 0 | Follow_behind | Following the lead vehicle at roughly steady speed and gap |
| 1 | Slow_down | Losing speed |
| 2 | Catch_up | Closing the gap to the lead vehicle |
| 3 | Speed_up | Gaining speed |
| 4 | Fall_behind | Gap to the lead vehicle opening up |
| 5 | Hold_speed | Holding speed, with a smaller/different signature than Follow_behind |

Patterns are derived from **inflection points** in the four smoothed signals above: each
vehicle's trajectory is segmented at points where velocity, acceleration, time headway, or
relative speed changes direction, then each resulting segment is classified into one of the 6
patterns by a fixed decision rule (velocity trend first, falling back to distance trend, then
acceleration trend, for segments with no clear speed trend). Across all 67,037 segments, pattern
frequency is: Follow_behind 19,063, Fall_behind 11,518, Speed_up 9,827, Catch_up 9,208,
Slow_down 8,896, Hold_speed 8,525. Segment duration ranges 1–239 frames (mean 47.9, median 41).

## How to access and use this data

All files are plain CSV — usable from Python, R, MATLAB, Excel (the two large trajectory files
exceed Excel's ~1M row limit, so use Python/R/a database for those), or any tool that reads CSV.

### Python / pandas

```python
import pandas as pd

# Everything for one driver in a single call:
df = pd.read_csv("trajectories_labeled_all_vehicles.csv")
driver = df[df["VehicleID"] == 68]

print(driver[["FrameID", "timestamp", "smooth_v_follow", "pattern_label", "pattern_duration"]])

# Segment-level summary for that driver:
print(driver.drop_duplicates("segment_index")[
    ["segment_index", "pattern_label", "pattern_start_frame", "pattern_end_frame", "pattern_duration"]
])
```

If you'd rather work with the smaller normalized files:

```python
traj = pd.read_csv("trajectories_all_vehicles.csv")
seg = pd.read_csv("segments_all_vehicles.csv")

merged = traj.merge(seg, on=["VehicleID", "segment_index"], how="left")
```

Loading the two large files (`trajectories_labeled_all_vehicles.csv`,
`trajectories_all_vehicles.csv`) whole can take a few seconds and a few hundred MB of RAM; if you
only need a handful of vehicles, filter with `usecols=` and chunked reading
(`pd.read_csv(..., chunksize=100_000)`), or ask to have these re-exported as Parquet, which loads
much faster and compresses considerably better than CSV.

### Reconstructing per-driver label + duration sequences (as used in the model)

```python
labels = pd.read_csv("labeled_data_all_vehicles.csv")
row = labels[labels["VehicleID"] == 68].iloc[0]

label_encoding = {"Follow_behind": 0, "Slow_down": 1, "Catch_up": 2,
                   "Speed_up": 3, "Fall_behind": 4, "Hold_speed": 5}

seq_labels = eval(row["Label"])   # e.g. ['Fall_behind', 'Speed_up', 'Fall_behind', ...]
seq_points = eval(row["Point"])   # e.g. [0, 64, 100, 128, 162, 199]

encoded_labels = [label_encoding[l] for l in seq_labels[1:]]
durations = [seq_points[i + 1] - seq_points[i] for i in range(len(seq_points) - 1)]
```

## License

- **Data** (all CSV files, wherever hosted): [CC BY 4.0](DATA_LICENSE.md) — free to
  share and adapt, with attribution.
- **Code** (the `code/` folder): [MIT License](LICENSE).

## Provenance

- Source data: `trainHH_data.csv` (Lyft Level 5 dataset, HV-HV car-following pairs), 16,829
  unique vehicles, 3,227,190 total frames.
- Labelling method: a rule-based segmentation algorithm applied to each vehicle's smoothed
  velocity, acceleration, time headway, and relative speed; see
  `code/build_full_trajectory_dataset.py` for the exact implementation used to generate every
  file in this folder.
- Generated: September 2026.

## Questions / regenerating this dataset

Run `code/build_full_trajectory_dataset.py` against `trainHH_data.csv` to regenerate
`labeled_data_all_vehicles.csv`, `segments_all_vehicles.csv`, and `trajectories_all_vehicles.csv`
from scratch (takes well under two minutes for all 16,829 vehicles). The merged
`trajectories_labeled_all_vehicles.csv` is then a straightforward pandas merge of those three
plus the source file's `timestamp` column, on (`VehicleID`, `segment_index`) and
(`VehicleID`, `FrameID`) respectively.
