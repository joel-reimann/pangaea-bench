print("[FORENSIC] Using agbd.py from:", __file__, flush=True)
import logging
logging.warning(f"[FORENSIC] Using agbd.py from: {__file__}")

# import logging
import os
import h5py
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from os.path import join, exists
from datetime import timedelta
from collections import Counter # <<< ADD THIS IMPORT
import glob


# Assuming pangaea.datasets.base defines RawGeoFMDataset correctly
from pangaea.datasets.base import RawGeoFMDataset

# --- Constants (from AGBD dataset.py) ---
NODATAVALS = {'S2_bands': 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}
# Define the 6 Prithvi bands (Sentinel-2 L2A) required
PRITHVI_BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

# --- Helper Functions (Adapted from AGBD dataset.py) ---

def initialize_index_sample_based(fnames, mode, path_mapping, path_h5):
    """
    Initializes index mapping files/tiles to sample counts for the specified mode.
    Correctly uses a mapping where keys are 'train'/'val'/'test' and values are lists of tile names.
    """
    index = {}
    total_samples = 0
    tile_lengths = {}

    logging.info(f"Initializing sample-based index for mode '{mode}'...")

    # --- Get the set of allowed tile names for the requested mode ---
    if mode not in path_mapping:
        logging.error(f"Mode '{mode}' not found as a key in the mapping file. Available keys: {list(path_mapping.keys())}")
        return index, 0 # Return empty index and 0 length

    # Convert tile names (which might be np.str_) to regular strings and put in a set
    try:
        allowed_tiles_for_mode = set(str(tile_name) for tile_name in path_mapping[mode])
        logging.debug(f"  Found {len(allowed_tiles_for_mode)} allowed tiles for mode '{mode}'. Sample: {list(allowed_tiles_for_mode)[:5]}")
        if not allowed_tiles_for_mode:
             logging.warning(f"  The list of tiles for mode '{mode}' in the mapping file is empty.")
             # Continue processing files, but likely won't find matches
    except Exception as e:
        logging.error(f"Error processing tile list for mode '{mode}' from mapping file: {e}")
        return index, 0
    # --- End getting allowed tiles ---

    for fname in fnames:
        fpath = join(path_h5, fname)
        if not exists(fpath):
            logging.warning(f"HDF5 file {fpath} not found, skipping.")
            continue

        logging.debug(f"Processing file: {fname}")
        try:
            with h5py.File(fpath, 'r') as f:
                tiles_in_file = list(f.keys())
                valid_tiles_in_file = {} # Changed variable name for clarity
                logging.debug(f"  Tiles found in file: {tiles_in_file[:10]}...")
                for tile in tiles_in_file:
                    # Use the full tile name from HDF5 as the key for lookup
                    # Assuming HDF5 keys directly match the names in the mapping list
                    tile_key_for_lookup = str(tile) # Ensure it's a string
                    logging.debug(f"    Checking tile: {tile_key_for_lookup}")

                    # --- Check if this tile is in the allowed set for the mode ---
                    if tile_key_for_lookup in allowed_tiles_for_mode:
                        logging.debug(f"      Tile '{tile_key_for_lookup}' is allowed for mode '{mode}'. Checking sample length...")
                        try:
                            agbd_path = f"{tile}/GEDI/agbd" # Use original tile name for HDF5 access
                            if agbd_path in f:
                                tile_len = len(f[agbd_path])
                                logging.debug(f"        Found '{agbd_path}'. Length: {tile_len}")
                                if tile_len > 0:
                                    valid_tiles_in_file[tile] = tile_len # Store original tile name and length
                                    total_samples += tile_len
                                    tile_lengths[tile] = tile_len
                                    logging.debug(f"          Tile added to index for mode '{mode}'. Current total samples: {total_samples}")
                                else:
                                    logging.debug(f"        Tile {tile} in {fname} has 0 samples, skipping.")
                            else:
                                logging.warning(f"        Dataset path '{agbd_path}' not found in tile {tile}, skipping length check.")
                        except Exception as e:
                            logging.warning(f"        Error checking length for tile {tile} in {fname}: {e}, skipping.")
                    else:
                        logging.debug(f"      Tile '{tile_key_for_lookup}' is NOT in the allowed list for mode '{mode}'. Skipping.")
                    # --- End check ---

                if valid_tiles_in_file:
                    index[fname] = valid_tiles_in_file
        except Exception as e:
            logging.error(f"Error processing HDF5 file {fpath} for index: {e}")

    logging.info(f"Index initialization complete for mode '{mode}'. Total samples: {total_samples}")
    # Add a check here before returning, similar to the one in __init__
    if total_samples == 0:
        logging.warning(f"Index initialization for mode '{mode}' resulted in 0 total samples. Double-check mapping file contents and HDF5 file paths/contents.")

    return index, total_samples


def find_sample_index(index_structure, sample_n):
    """
    Finds the file name, tile name, and index within the tile for a given overall sample index `sample_n`.
    Assumes index_structure maps fname -> {tile_name: tile_len}.
    """
    current_sample_count = 0

    for fname, tiles in index_structure.items():
        # if fname == '_tile_lengths': continue # Example if using metadata key
        for tile_name, tile_len in tiles.items():
            if sample_n < current_sample_count + tile_len:
                # Found the correct file and tile
                sample_idx_in_tile = sample_n - current_sample_count
                return fname, tile_name, sample_idx_in_tile
            current_sample_count += tile_len

    # If we reach here, index n was out of bounds
    raise IndexError(f"Sample index {sample_n} is out of calculated range {current_sample_count}")


def normalize_data(data, norm_values, norm_strat, nodata = None):
    """ Normalize data using either min-max or percentile normalization. Adapted from AGBD dataset.py """
    if not isinstance(norm_values, dict) or ('min' not in norm_values and '1%' not in norm_values):
         logging.warning(f"Invalid norm_values format for normalization: {norm_values}. Skipping normalization.")
         return data

    data_float = data.astype(np.float32) # Work with float copy
    valid_mask = None
    if nodata is not None:
        valid_mask = (data_float != nodata)
        if not np.any(valid_mask): return data_float # Avoid division by zero if all nodata

    if norm_strat == 'min_max':
        min_val, max_val = norm_values.get('min', 0), norm_values.get('max', 1)
        range_val = max_val - min_val
        if range_val == 0: return data_float # Avoid division by zero
        if valid_mask is not None:
            data_float[valid_mask] = (data_float[valid_mask] - min_val) / range_val
        else:
            data_float = (data_float - min_val) / range_val
    elif norm_strat == 'pct':
        pct_1, pct_99 = norm_values.get('1%', 0), norm_values.get('99%', 1)
        range_val = pct_99 - pct_1
        if range_val == 0: return data_float # Avoid division by zero
        if valid_mask is not None:
            data_float[valid_mask] = (data_float[valid_mask] - pct_1) / range_val
        else:
            data_float = (data_float - pct_1) / range_val
        # Clip to [0, 1] after percentile normalization
        data_float = np.clip(data_float, 0, 1)
    else:
        raise ValueError(f"Normalization strategy must be 'min_max' or 'pct', got {norm_strat}")
    return data_float

def normalize_bands(bands_data, norm_values_dict, band_order, norm_strat, nodata = None):
    """ Normalize multi-band data using normalize_data for each band. Adapted from AGBD dataset.py """
    normalized_bands = np.zeros_like(bands_data, dtype=np.float32)
    if bands_data.shape[-1] != len(band_order):
         raise ValueError(f"Number of bands in data ({bands_data.shape[-1]}) does not match band_order length ({len(band_order)})")

    for i, band_name in enumerate(band_order):
        if band_name in norm_values_dict:
            normalized_bands[..., i] = normalize_data(bands_data[..., i], norm_values_dict[band_name], norm_strat, nodata)
        else:
            logging.warning(f"Normalization values not found for band: {band_name}. Skipping normalization.")
            normalized_bands[..., i] = bands_data[..., i].astype(np.float32)
    return normalized_bands


class AGBDDataset(RawGeoFMDataset):
    """
    Pangaea-compatible Dataset class for AGBD, loading data from HDF5 files
    using the structure and helper files from the original AGBD repository.
    Loads only Sentinel-2 optical bands for Prithvi and the AGBD target.
    Uses sample-based indexing.
    """
    # --- Constants for additional bands/modalities ---
    S1_BANDS = ['VV', 'VH']
    ALOS_BANDS = ['HH', 'HV']
    DEM_BAND = 'DEM'
    LC_BAND = 'LC'
    CH_BAND = 'CH'
    # Add to class as static/class variables for clarity

    def __init__(self, root_path, split, bands, img_size, # From Pangaea config
                 # Required by RawGeoFMDataset base
                 dataset_name, multi_modal, multi_temporal, classes, num_classes,
                 ignore_index, distribution, data_mean, data_std, data_min, data_max,
                 download_url, auto_download,
                 # Specific to AGBD logic (add to config)
                 years=[2019, 2020], version=4, norm_strat='pct',
                 debug=False,
                 h5_path=None, # NEW: allow override
                 h5_pattern=None, # NEW: allow pattern override
                 **kwargs): # Catch any other args

        # Call Pangaea base class init FIRST
        super().__init__(
            root_path=root_path, split=split, bands=bands, img_size=img_size,
            dataset_name=dataset_name, multi_modal=multi_modal, multi_temporal=multi_temporal,
            classes=classes, num_classes=num_classes, ignore_index=ignore_index,
            distribution=distribution, data_mean=data_mean, data_std=data_std,
            data_min=data_min, data_max=data_max, download_url=download_url,
            auto_download=auto_download, **kwargs
        )

        # --- AGBD Specific Init Logic ---
        self.h5_path = h5_path if h5_path is not None else self.root_path
        self.norm_path = self.root_path # Assuming PKL files are directly in root_path
        # self.mapping_path = self.root_path # Assuming biomes_splits pkl is in root_path #debug to test debug on euler
        self.mapping_path = '/cluster/home/reimannj/' #debug to test debug on euler

        self.mode = self.split # Map Pangaea split to AGBD mode ('train', 'val', 'test')
        self.years = years
        self.version = version
        self.norm_strat = norm_strat
        self.debug = debug
        self.h5_pattern = h5_pattern or f"*v{self.version}_*-20.h5"

        # Load mapping file (defines train/val/test split based on tile names)
        mapping_file = join(self.mapping_path, 'biomes_splits_to_name.pkl')
        if not exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        try:
            with open(mapping_file, 'rb') as f:
                self.mapping = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading mapping file {mapping_file}: {e}")
            raise

        # Get HDF5 filenames based on pattern
        all_files = sorted(glob.glob(os.path.join(self.h5_path, self.h5_pattern)))
        self.fnames = [os.path.basename(f) for f in all_files]
        if self.debug:
            self.fnames = self.fnames[:4]  # Only a few files for debug

        # Check existence of expected HDF5 files (optional, for early warning)
        for fname in self.fnames:
             fpath_check = join(self.h5_path, fname)
             if not exists(fpath_check):
                 # This might be okay if the file simply has no tiles for the current split
                 logging.debug(f"Expected HDF5 file not found: {fpath_check}")

        # Initialize index (using sample-based helper function)
        self.index, self.length = initialize_index_sample_based(self.fnames, self.mode, self.mapping, self.h5_path)
        if self.length == 0:
             # This is a critical error - no data found for this split
             raise ValueError(f"Dataset split '{self.mode}' resulted in 0 samples. Check mapping file, HDF5 contents, and root_path.")

        # Load normalization values (MUST match AGBD official logic)
        norm_file = join(self.norm_path, f"statistics_subset_2019-2020-v{self.version}_new.pkl")
        # test the other file
        # norm_file = join(self.norm_path, f"statistics_subset.pkl")

        if not exists(norm_file):
            raise FileNotFoundError(f"Normalization file not found: {norm_file}")
        try:
            with open(norm_file, mode='rb') as f:
                self.norm_values = pickle.load(f)
            logging.info(f"Loaded normalization file: {norm_file} (keys: {list(self.norm_values.keys())})")
        except Exception as e:
            logging.error(f"Error loading normalization file {norm_file}: {e}")
            raise

        # --- Overwrite all normalization attributes with values from .pkl file ---
        # IMPORTANT: For AGBD, all normalization values (mean, std, min, max, etc.) are loaded from the .pkl file.
        # The config values (data_mean, data_std, data_min, data_max) are ignored and replaced here.
        # This ensures correct normalization and prevents bugs if config values are stale or dummy.
        #
        # The .pkl file (self.norm_values) contains per-band/group statistics for all relevant modalities.
        # We extract the relevant stats for each modality and set the attributes accordingly.

        def _extract_stat_from_pkl(norm_values, stat):
            # Helper to extract a dict of modality -> list of stat values
            out = {}
            # Sentinel-2 bands (optical)
            if 'S2_bands' in norm_values:
                out['optical'] = [norm_values['S2_bands'][b][stat] for b in self.bands.get('optical', []) if b in norm_values['S2_bands']]
            # Add other modalities as needed (ALOS, CH, DEM, etc.)
            if 'ALOS_bands' in norm_values:
                out['alos'] = [norm_values['ALOS_bands'][b][stat] for b in self.ALOS_BANDS if b in norm_values['ALOS_bands']]
            if 'CH' in norm_values and stat in norm_values['CH']:
                out['ch'] = [norm_values['CH'][stat]]
            if 'DEM' in norm_values and stat in norm_values['DEM']:
                out['dem'] = [norm_values['DEM'][stat]]
            # Add more as needed
            return out

        # Overwrite normalization attributes
        self.data_mean = _extract_stat_from_pkl(self.norm_values, 'mean')
        self.data_std  = _extract_stat_from_pkl(self.norm_values, 'std')
        self.data_min  = _extract_stat_from_pkl(self.norm_values, 'min')
        self.data_max  = _extract_stat_from_pkl(self.norm_values, 'max')
        # Optionally, add percentiles if needed:
        # self.data_p1 = _extract_stat_from_pkl(self.norm_values, '1%')
        # self.data_p99 = _extract_stat_from_pkl(self.norm_values, '99%')

        # --- Runtime warning if dummy config values are still present ---
        def _is_dummy(vals):
            # Detects if a normalization dict is still set to dummy values (e.g., all 0.5 or 0/1)
            if not isinstance(vals, dict):
                return False
            for v in vals.values():
                if isinstance(v, (list, tuple)) and all(x in (0.0, 0.5, 1.0) for x in v):
                    return True
            return False
        if any(_is_dummy(x) for x in [data_mean, data_std, data_min, data_max]):
            logging.warning("[AGBDDataset] Dummy normalization values from config detected and ignored. All normalization is loaded from the .pkl file.")

        # --- Docstring for maintainers ---
        self.__doc__ = (self.__doc__ or "") + "\n\nNOTE: For AGBD, all normalization values are loaded from the .pkl file.\nConfig values for normalization are ignored and replaced at runtime.\nDo not rely on config normalization values for this dataset."

        # Open HDF5 handles (potential issue with num_workers > 0 in DataLoader)
        # Only open files that are actually part of the index for this split
        self.handles = {}
        files_in_index = set(self.index.keys()) # Get unique filenames from the generated index
        for fname in files_in_index:
            fpath = join(self.h5_path, fname)
            # No need to check exists(fpath) here, initialize_index already did implicitly
            try:
                 self.handles[fname] = h5py.File(fpath, 'r')
                 logging.debug(f"Opened HDF5 handle for {fname}")
            except Exception as e:
                 logging.error(f"Failed to open HDF5 file {fpath} needed by index: {e}")
                 # Clean up already opened handles before raising
                 for handle in self.handles.values(): handle.close()
                 raise

        # Hardcoded patch size info based on AGBD dataset.py inspection
        # Assumes HDF5 stores 25x25 patches centered at index 12
        self.h5_internal_size = 25
        self.h5_patch_center = 12
        self.h5_patch_window = self.h5_internal_size // 2 # = 12

        # Check if Pangaea img_size matches HDF5 internal size
        if self.img_size != self.h5_internal_size:
             logging.warning(
                 f"Pangaea config img_size ({self.img_size}) differs from AGBD HDF5 internal patch size ({self.h5_internal_size}). "
                 f"Loading full {self.h5_internal_size}x{self.h5_internal_size} patch. Ensure Pangaea preprocessor handles resizing if needed."
             )
             # We will load the full 25x25 patch regardless of self.img_size

        logging.info(f"Initialized AGBDDataset split '{self.split}'. Total samples: {self.length}")
        # --- End AGBD Specific Init ---

    def _get_band_indices(self, band_list, h5_band_order):
        """Helper to get indices of requested bands in HDF5 band order."""
        indices = []
        for band in band_list:
            if band in h5_band_order:
                indices.append(h5_band_order.index(band))
            else:
                logging.warning(f"Band {band} not found in HDF5 band order {h5_band_order}")
        return indices

    # --- Helper: Modular band loading and normalization ---
    def _load_and_normalize(self, f, tile_name, group, band_names, norm_key, norm_strat, nodata=None, postprocess=None):
        """Load and normalize a set of bands from HDF5, return (H, W, C) float32 array."""
        try:
            # Get band order from HDF5 attributes (cache per group)
            cache_attr = f'_order_in_h5_{group}'
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, list(f[tile_name][group].attrs['order']))
            order_in_h5 = getattr(self, cache_attr)
            indices = [order_in_h5.index(b) for b in band_names]
            loaded_band_names = [order_in_h5[i] for i in indices]
            # Extract patch (H, W, C)
            arr = f[tile_name][group][self._idx_in_tile, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1, :]
            arr = arr[:, :, indices].astype(np.float32)
            if postprocess:
                arr = postprocess(arr)
            arr = normalize_bands(arr, self.norm_values[norm_key], loaded_band_names, norm_strat, nodata)
            return arr
        except Exception as e:
            logging.error(f"Error loading group {group} bands {band_names} from tile {tile_name}: {e}")
            raise

    def _load_single_band(self, f, tile_name, group, norm_key, norm_strat, nodata=None, postprocess=None):
        """Load and normalize a single-band raster (DEM, LC, CH). Returns (H, W, 1) float32 array."""
        try:
            arr = f[tile_name][group][self._idx_in_tile, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1]
            arr = arr.astype(np.float32)
            if postprocess:
                arr = postprocess(arr)
            arr = normalize_data(arr, self.norm_values[norm_key], norm_strat, nodata)
            arr = arr[..., None]  # (H, W, 1)
            return arr
        except Exception as e:
            logging.error(f"Error loading group {group} from tile {tile_name}: {e}")
            raise

    def _load_metadata(self, f, tile_name, idx_in_tile):
        """Load metadata fields (lat, lon, doy, etc) as a dict."""
        try:
            meta = {}
            meta_fields = ['lat', 'lon', 'doy', 'year', 'biome', 'shot_number']
            for field in meta_fields:
                if field in f[tile_name]['GEDI']:
                    meta[field] = f[tile_name]['GEDI'][field][idx_in_tile]
            return meta
        except Exception as e:
            logging.error(f"Error loading metadata for tile {tile_name}: {e}")
            return {}

    def __getitem__(self, n):
        """Loads the n-th sample (0-based index) of the dataset split."""
        if n < 0 or n >= self.length:
            raise IndexError(f"Index {n} out of range for dataset length {self.length}")

        # Find the file name, tile name, and sample index within the tile
        try:
            file_name, tile_name, idx_in_tile = find_sample_index(self.index, n)
        except IndexError as e:
             logging.error(f"Error finding sample index for master index {n}: {e}")
             raise
        except Exception as e:
             logging.error(f"Unexpected error in find_sample_index for master index {n}: {e}")
             raise


        # Get the file handle
        if file_name not in self.handles:
            logging.error(f"HDF5 handle for {file_name} not found (needed for master index {n}). This should not happen.")
            raise RuntimeError(f"HDF5 handle for {file_name} not found.")
        f = self.handles[file_name]

        try:
            # --- Save idx_in_tile for helpers ---
            self._idx_in_tile = idx_in_tile

            # --- S2 (Prithvi) ---
            s2_bands = PRITHVI_BANDS
            # Forensic: load raw patch before any scaling/normalization
            try:
                file_handle = f
                raw_patch = file_handle[tile_name]['S2_bands'][idx_in_tile, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1, self.h5_patch_center - self.h5_patch_window : self.h5_patch_center + self.h5_patch_window + 1, :]
                raw_patch = np.array(raw_patch)
                print(f"[FORENSIC] S2 raw patch (before scaling) min: {raw_patch.min():.4f}, max: {raw_patch.max():.4f}, mean: {raw_patch.mean():.4f}, shape: {raw_patch.shape}")
                scaled_patch = raw_patch / 10000.0
                print(f"[FORENSIC] S2 patch after /10000 scaling min: {scaled_patch.min():.4f}, max: {scaled_patch.max():.4f}, mean: {scaled_patch.mean():.4f}")
                # Print normalization stats for each band
                for b in s2_bands:
                    norm_stats = self.norm_values['S2_bands'][b]
                    print(f"[FORENSIC] Norm stats for {b}: {norm_stats}")
            except Exception as e:
                print(f"[FORENSIC] Error in S2 forensic logging: {e}")
            s2_post = lambda arr: arr / 10000.0  # DN to reflectance
            s2_arr = self._load_and_normalize(f, tile_name, 'S2_bands', s2_bands, 'S2_bands', self.norm_strat, NODATAVALS['S2_bands'], postprocess=s2_post)
            print(f"[FORENSIC] S2 patch after normalization min: {s2_arr.min():.4f}, max: {s2_arr.max():.4f}, mean: {s2_arr.mean():.4f}, shape: {s2_arr.shape}")
            s2_tensor = torch.from_numpy(s2_arr.copy()).permute(2, 0, 1).unsqueeze(1)  # (C, T=1, H, W)

            # --- Helper to check if group is a dataset ---
            def _is_dataset(f, tile_name, group):
                try:
                    return group in f[tile_name] and isinstance(f[tile_name][group], h5py.Dataset)
                except Exception:
                    return False

            # --- S1 ---
            s1_tensor = None
            if _is_dataset(f, tile_name, 'S1_bands') and 'S1_bands' in self.norm_values:
                s1_bands = list(f[tile_name]['S1_bands'].attrs['order'])
                s1_arr = self._load_and_normalize(f, tile_name, 'S1_bands', s1_bands, 'S1_bands', self.norm_strat, NODATAVALS.get('S1_bands', 0))
                s1_tensor = torch.from_numpy(s1_arr.copy()).permute(2, 0, 1).unsqueeze(1)
            elif 'S1_bands' in f[tile_name]:
                logging.warning(f"S1_bands group present in HDF5 but not a dataset or missing normalization stats. Skipping S1.")

            # --- ALOS ---
            alos_tensor = None
            if _is_dataset(f, tile_name, 'ALOS_bands') and 'ALOS_bands' in self.norm_values:
                alos_bands = list(f[tile_name]['ALOS_bands'].attrs['order'])
                alos_arr = self._load_and_normalize(f, tile_name, 'ALOS_bands', alos_bands, 'ALOS_bands', self.norm_strat, NODATAVALS['ALOS_bands'])
                alos_tensor = torch.from_numpy(alos_arr.copy()).permute(2, 0, 1).unsqueeze(1)
            elif 'ALOS_bands' in f[tile_name]:
                logging.warning(f"ALOS_bands group present in HDF5 but not a dataset or missing normalization stats. Skipping ALOS.")

            # --- DEM ---
            dem_tensor = None
            if _is_dataset(f, tile_name, 'DEM') and 'DEM' in self.norm_values:
                dem_arr = self._load_single_band(f, tile_name, 'DEM', 'DEM', self.norm_strat, NODATAVALS['DEM'])
                dem_tensor = torch.from_numpy(dem_arr.copy()).permute(2, 0, 1).unsqueeze(1)
            elif 'DEM' in f[tile_name]:
                logging.warning(f"DEM group present in HDF5 but not a dataset or missing normalization stats. Skipping DEM.")

            # --- LC ---
            lc_tensor = None
            if _is_dataset(f, tile_name, 'LC') and 'LC' in self.norm_values:
                lc_arr = self._load_single_band(f, tile_name, 'LC', 'LC', self.norm_strat, NODATAVALS['LC'])
                lc_tensor = torch.from_numpy(lc_arr.copy()).permute(2, 0, 1).unsqueeze(1)
            elif 'LC' in f[tile_name]:
                logging.warning(f"LC group present in HDF5 but not a dataset or missing normalization stats. Skipping LC.")

            # --- CH ---
            ch_tensor = None
            if _is_dataset(f, tile_name, 'CH') and 'CH' in self.norm_values:
                ch_arr = self._load_single_band(f, tile_name, 'CH', 'CH', self.norm_strat, NODATAVALS['CH'])
                ch_tensor = torch.from_numpy(ch_arr.copy()).permute(2, 0, 1).unsqueeze(1)
            elif 'CH' in f[tile_name]:
                logging.warning(f"CH group present in HDF5 but not a dataset or missing normalization stats. Skipping CH.")

            # --- Target (AGBD) ---
            agbd_scalar = f[tile_name]['GEDI']['agbd'][idx_in_tile]
            _, _, height, width = s2_tensor.shape
            agbd_tensor = torch.full((height, width), agbd_scalar, dtype=torch.float32)

            # --- Metadata ---
            meta = {
                "master_index": n, "h5_file": file_name, "tile_name": tile_name,
                "h5_index": idx_in_tile, "split": self.split,
            }
            meta.update(self._load_metadata(f, tile_name, idx_in_tile))

            # --- Compose output dict ---
            image = {"optical": s2_tensor}
            if s1_tensor is not None:
                image["s1"] = s1_tensor
            if alos_tensor is not None:
                image["alos"] = alos_tensor
            if dem_tensor is not None:
                image["dem"] = dem_tensor
            if lc_tensor is not None:
                image["lc"] = lc_tensor
            if ch_tensor is not None:
                image["ch"] = ch_tensor

            return {"image": image, "target": agbd_tensor, "meta": meta}

        except Exception as e:
            logging.error(f"Error reading data for master index {n} (HDF5: {file_name}/{tile_name}[{idx_in_tile}]): {e}")
            raise

    def download(self, *args, **kwargs):
        # No-op: AGBD data must be pre-downloaded
        return

    def __del__(self):
        """Closes all open HDF5 file handles."""
        logging.info(f"Closing HDF5 handles for AGBDDataset split '{self.split}'...")
        closed_count = 0
        for fname, handle in self.handles.items():
            try:
                if handle: # Check if handle exists and is open
                     handle.close()
                     closed_count += 1
            except Exception as e:
                logging.warning(f"Exception while closing HDF5 handle for {fname}: {e}")
        self.handles.clear()
        logging.info(f"Closed {closed_count} HDF5 handles.")

    def __len__(self):
        """Returns the total number of samples in this split."""
        return self.length
