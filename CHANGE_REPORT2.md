# Change Report: ivi-imaging originals vs modified

Date: 2026-03-09

Scope:
- Original code: `ivi-imaging/src/CellAnalyzer.py` and `ivi-imaging/notebooks/vetsuisse25_ALI_Project.ipynb`
- Modified code: `modified/src/CellAnalyzer_2.py` and `modified/notebooks/vetsuisse25_ALI_Project_2.ipynb`

Summary of intent
- The modifications switch from bulk loading to sequential, per-image loading and projection to avoid memory issues when processing many images.

## Major Change
### ivi-imaging/src/CellAnalyzer.py -> modified/src/CellAnalyzer_2.py
1. Loading pipeline replaced with sequential projection
   - Original `load` and `read_data` exist at `ivi-imaging/src/CellAnalyzer.py:101` and `ivi-imaging/src/CellAnalyzer.py:154`; these are commented out in the modified file (see commented versions starting at `modified/src/CellAnalyzer_2.py:338` and `modified/src/CellAnalyzer_2.py:392`).
   - Original `create_projections` exists at `ivi-imaging/src/CellAnalyzer.py:257` and is commented out in modified (`modified/src/CellAnalyzer_2.py:629`).
   - New `load_and_project` added at `modified/src/CellAnalyzer_2.py:100`, which sequentially loads one image at a time and builds `samples_df` and `projections` (`modified/src/CellAnalyzer_2.py:209`, `modified/src/CellAnalyzer_2.py:227`).
   - File extension changed from `.nd2` in the original parser (`ivi-imaging/src/CellAnalyzer.py:188` and `ivi-imaging/src/CellAnalyzer.py:195`) to `.ome.tif` in the modified loader (`modified/src/CellAnalyzer_2.py:114` and `modified/src/CellAnalyzer_2.py:119`).

## Additional Changes
### ivi-imaging/src/CellAnalyzer.py -> modified/src/CellAnalyzer_2.py

1. Imports and module-level side effects
   - PIL import trimmed (removed `ImageDraw`, `ImageFont`) at `ivi-imaging/src/CellAnalyzer.py:1` vs `modified/src/CellAnalyzer_2.py:1`. This aligns with removal of scale-bar drawing.
   - Cellpose import now includes `models` at `modified/src/CellAnalyzer_2.py:4` (not present at `ivi-imaging/src/CellAnalyzer.py:4`).
   - `torch` import added at `modified/src/CellAnalyzer_2.py:14` (not present in original imports).
   - Module import now prints a message at `modified/src/CellAnalyzer_2.py:17` (no equivalent in original).

2. Class state and serialization changes
   - `signal_mode` default changed from a dict to a string: `ivi-imaging/src/CellAnalyzer.py:38` vs `modified/src/CellAnalyzer_2.py:40`.
   - `cfg_df` attribute removed: `ivi-imaging/src/CellAnalyzer.py:41` has it; no equivalent in modified.
   - Saving `metadata_cfg.csv` removed: `ivi-imaging/src/CellAnalyzer.py:72` is gone in modified (see `modified/src/CellAnalyzer_2.py:69`).
   - `cfg_df` removed from pickled payload: `ivi-imaging/src/CellAnalyzer.py:95` vs `modified/src/CellAnalyzer_2.py:93`.


3. Segmentation changes
   - Cellpose eval now sets `batch_size=4` at `modified/src/CellAnalyzer_2.py:757`; original used default batch size at `ivi-imaging/src/CellAnalyzer.py:385`.
   - Mask dtype forced to `np.int32` at `modified/src/CellAnalyzer_2.py:777`; original used conditional dtype at `ivi-imaging/src/CellAnalyzer.py:397`.
   - Per-cell logging added in `create_cells_df` at `modified/src/CellAnalyzer_2.py:854` and `modified/src/CellAnalyzer_2.py:856`.

4. Signal calculation behavior changed
   - `calculate_single_cell_signal` removed (original at `ivi-imaging/src/CellAnalyzer.py:577`), and logic is now inlined in `calculate_cell_signals` (`modified/src/CellAnalyzer_2.py:972`).
   - Column naming changed: original writes `name + "_signal"` and `name + "_signal_log10"` at `ivi-imaging/src/CellAnalyzer.py:666` and `ivi-imaging/src/CellAnalyzer.py:669`; modified writes `name + "_" + mode` and `name + "_" + mode + "_log10"` at `modified/src/CellAnalyzer_2.py:1052` and `modified/src/CellAnalyzer_2.py:1054`.
   - `signal_mode` is now a single string set at `modified/src/CellAnalyzer_2.py:1000`; original stored per-channel modes at `ivi-imaging/src/CellAnalyzer.py:615`.

5. Binning API and behavior changes
   - `bin_single_cell_signal` replaced by `bin_cell_signal` with a new signature at `modified/src/CellAnalyzer_2.py:1116`; original method at `ivi-imaging/src/CellAnalyzer.py:858`.
   - `bin_cell_signals` (multi-signal helper) removed; original at `ivi-imaging/src/CellAnalyzer.py:961`.
   - Binning now uses column name based on `signal` and `self.signal_mode` (`modified/src/CellAnalyzer_2.py:1139`) instead of `signal + "_signal"` (`ivi-imaging/src/CellAnalyzer.py:888`).
   - Default bin column name now uses `col_name` or `signal` (`modified/src/CellAnalyzer_2.py:1168`) instead of `signal + "_bin"` (`ivi-imaging/src/CellAnalyzer.py:921`).
   - Bin masks now use `float32` and are built by accumulation (`modified/src/CellAnalyzer_2.py:1183` and `modified/src/CellAnalyzer_2.py:1189`), instead of `uint16` direct assignment (`ivi-imaging/src/CellAnalyzer.py:946` and `ivi-imaging/src/CellAnalyzer.py:953`).
   - Threshold metadata is now stored directly in `cells_df` (`modified/src/CellAnalyzer_2.py:1177`), whereas the original stored metadata in `cfg_df` (`ivi-imaging/src/CellAnalyzer.py:929`).

6. Scale-bar functionality removed
   - `save_segmentation_imgs` no longer accepts `scale_bar_px`/`scale_bar_um` (original signature at `ivi-imaging/src/CellAnalyzer.py:478`, modified at `modified/src/CellAnalyzer_2.py:879`).
   - `save_signal_masks`, `save_bin_masks`, `save_population_masks` signatures removed scale-bar args (original at `ivi-imaging/src/CellAnalyzer.py:806`, `ivi-imaging/src/CellAnalyzer.py:1010`, `ivi-imaging/src/CellAnalyzer.py:1104`; modified at `modified/src/CellAnalyzer_2.py:1069`, `modified/src/CellAnalyzer_2.py:1195`, `modified/src/CellAnalyzer_2.py:1281`).
   - `_add_scale_bar` function removed (original at `ivi-imaging/src/CellAnalyzer.py:1221`).

7. Population creation behavior changed
   - `create_populations` no longer auto-appends `_bin` to signal names. Original auto-suffix logic at `ivi-imaging/src/CellAnalyzer.py:1077`; modified expects exact column names at `modified/src/CellAnalyzer_2.py:1257`.

### ivi-imaging/notebooks/vetsuisse25_ALI_Project.ipynb -> modified/notebooks/vetsuisse25_ALI_Project_2.ipynb

Note: Notebook line references below correspond to a code-only extraction of notebook cells (to avoid output noise). Cell numbers are included for clarity.

1. Imports and environment setup
   - Original imports `CellAnalyzer` at line 16 in CELL 0; modified imports `CellAnalyzer_2` at line 20 in CELL 0.
   - Modified adds `Path` at line 14 and CUDA allocator env + `torch` at line 23 and line 25 in CELL 0.

2. Paths and initialization
   - Original uses Windows paths and loads an existing instance (`CellAnalyzer.load`) at line 27 in CELL 3.
   - Modified uses new Linux paths at line 31 and line 34 in CELL 1, and creates a new instance at line 37 in CELL 2.

3. Data loading and projections
   - Original calls `create_projections` at line 52 in CELL 7.
   - Modified uses `load_and_project` at line 41 in CELL 3.
   - Original references `ca.cfg_df` at line 55 in CELL 8; modified references `ca.samples_df` at line 44 in CELL 4.

4. Signal analysis flow simplified
   - Original calculates infection/goblet/cilia with custom dilation and modes at line 103, line 104, and line 105 in CELL 16.
   - Modified calculates only goblet/cilia mean at line 90 in CELL 12.
   - Original uses `prefix`-based aggregation and infection thresholding (line 115 in CELL 19 and line 118 in CELL 20). Modified removes those blocks and uses `condition` aggregation at line 97 in CELL 14.

5. Binning and population changes
   - Original uses `bin_cell_signals` at line 168 in CELL 25; modified uses `bin_cell_signal` per signal at line 138 and line 141 in CELL 20 and CELL 21.
   - Original renames bin columns at line 177 in CELL 28; modified does not rename bins.
   - Original creates populations with `infection/cilia/goblet` at line 203 in CELL 35; modified uses `cilia/goblet` at line 160 in CELL 26.
   - Original reports by `prefix` at line 239 in CELL 42; modified reports by `condition` at line 197 in CELL 33.

6. Additional condition summaries in modified notebook
   - Added WT/NT04/NT02 blocks at lines 176, 181, and 186 in CELL 30, CELL 31, and CELL 32 of the modified notebook.