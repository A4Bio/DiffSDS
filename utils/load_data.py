import pickle
import functools
import multiprocessing
import os
import os.path as osp
import glob
import logging
from pathlib import Path
from typing import *
# import torch.multiprocessing as mp
# mp.set_start_method("spawn")
import json
from matplotlib import pyplot as plt
import numpy as np
import lmdb
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from utils import beta_schedules
from utils import md5_all_py_files, wrapped_mean, modulo_with_wrapped_range, tolerant_comparison_check
from utils.angles_and_coords import (
    canonical_distances_and_dihedrals,
    EXHAUSTIVE_ANGLES,
    EXHAUSTIVE_DISTS,
    extract_backbone_coords,
    extract_backbone_seqs
)
from utils.nerf import TorchNERFBuilder

alphabet='ACDEFGHIKLMNPQRSTVWYU'
alphabet_map = {val:i for i,val in enumerate(alphabet)}

LOCAL_DATA_DIR = Path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)

CATH_DIR = LOCAL_DATA_DIR / "cath"
ALPHAFOLD_DIR = LOCAL_DATA_DIR / "alphafold"
PREPROCESSED_DIR = LOCAL_DATA_DIR / "cath" / "preprocessed_lmdb"
TRIM_STRATEGIES = Literal["leftalign", "randomcrop", "discard"]


class CathCanonicalAnglesDataset(Dataset):
    """
    Load in the dataset.

    All angles should be given between [-pi, pi]
    """

    feature_names = {
        "angles": [
            "0C:1N",
            "N:CA",
            "CA:C",
            "phi",
            "psi",
            "omega",
            "tau",
            "CA:C:1N",
            "C:1N:1CA",
        ],
        "coords": ["x", "y", "z"],
    }
    feature_is_angular = {
        "angles": [False, False, False, True, True, True, True, True, True],
        "coords": [False, False, False],
    }
    processed_path = PREPROCESSED_DIR

    def __init__(
        self,
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 512,
        min_length: int = 40,  # Set to 0 to disable
        trim_strategy: TRIM_STRATEGIES = "leftalign",
        zero_center: bool = True,  # Center the features to have 0 mean
        strict_test = False,
        use_cache: bool = True,  # Use/build cached computations of dihedrals and angles
    ) -> None:
        super().__init__()
        self.split = split
        self.trim_strategy = trim_strategy
        self.pad = pad
        self.min_length = min_length
        self.zero_center = zero_center
        self.strict_test = strict_test
        self.use_cache = use_cache
        self.split_name = split
        self.db = None
        self.means = None
        
        self.rng = np.random.default_rng(seed=6489)
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
            self._preprocess()
    
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            osp.join(self.processed_path, f"{self.split_name}.lmdb"),
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            self.keys.remove("means".encode())
        
        if self.strict_test and self.split_name=="test":
            split = pd.read_csv("/gaozhangyang/experiments/ProreinBinder/data/cath/CAT_split_TMscore.csv")
            remove = split[split["test"]==-1]["domain"].tolist()
            remove = [one.encode() for one in remove]
            self.keys = list(set(self.keys) - set(remove))
        self.keys = sorted(self.keys)
        

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _preprocess(self):
        
        # gather files
        fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
        codebase_hash = md5_all_py_files(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.structures = self.__compute_featurization(fnames)
        # If specified, remove sequences shorter than min_length
        if self.min_length:
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["angles"].shape[0] >= self.min_length
            ]
            len_delta = orig_len - len(self.structures)
            logging.info(
                f"Removing structures shorter than {self.min_length} residues excludes {len_delta}/{orig_len} --> {len(self.structures)} sequences"
            )
        if self.trim_strategy == "discard":
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["angles"].shape[0] <= self.pad
            ]
            len_delta = orig_len - len(self.structures)
            logging.info(
                f"Removing structures longer than {self.pad} produces {orig_len} - {len_delta} = {len(self.structures)} sequences"
            )

        # a 80/10/10 split
        # topology split
        split_df = pd.read_csv("/gaozhangyang/experiments/ProreinBinder/data/cath/CAT_split_TMscore.csv")
        split_df = split_df.set_index("domain")

        train_idx = []
        valid_idx = []
        test_idx = []
        for idx, struct in enumerate(self.structures):
            domain = struct["fname"].split("/")[-1]
            if split_df.loc[domain, "train"] == 1:
                train_idx.append(idx)
            elif split_df.loc[domain, "valid"] == 1:
                valid_idx.append(idx)
            elif split_df.loc[domain, "test"] == 1 or split_df.loc[domain, "test"] == -1:
                test_idx.append(idx)

        
        
        train_structures = [self.structures[i] for i in train_idx]
        valid_structures = [self.structures[i] for i in valid_idx]
        test_structures = [self.structures[i] for i in test_idx]
      

        # if given, zero center the features
        self.means = None
        if self.zero_center:
            # Note that these angles are not yet padded
            structures_concat = np.concatenate([s["angles"] for s in self.structures])
            self.means = wrapped_mean(structures_concat, axis=0)
            # Subtract the mean and perform modulo where values are radial
            logging.info(
                f"Offsetting features {self.feature_names['angles']} by means {self.means}"
            )

        # Aggregate lengths
        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )
        
        
        memory = {'train':10,  'valid':5, 'test':5}
        for split_name in ["train", "valid", "test"]:
            db = lmdb.open(
                osp.join(self.processed_path, f"{split_name}.lmdb"),
                map_size=memory[split_name]*(1024*1024*1024),   # 10GB
                create=True,
                subdir=False,
                readonly=False, 
            )
            
            if split_name == "train":
                data_list = train_structures
            if split_name == "valid":
                data_list = valid_structures
            if split_name == "test":
                data_list = test_structures
                
            with db.begin(write=True, buffers=True) as txn:
                for idx, data in enumerate(data_list):
                    txn.put(
                                key = str(data['fname'].split('/')[-1]).encode(),
                                value = pickle.dumps(data)
                            )
                txn.put(
                    key = 'means'.encode(),
                    value = pickle.dumps(self.means)
                )

            db.close()


    def __compute_featurization(
        self, fnames: Sequence[str]
    ) -> List[Dict[str, np.ndarray]]:
        """Get the featurization of the given fnames"""
        pfunc = functools.partial(
            canonical_distances_and_dihedrals,
            distances=EXHAUSTIVE_DISTS,
            angles=EXHAUSTIVE_ANGLES,
        )
        coords_pfunc = functools.partial(extract_backbone_coords, atoms=["N", "CA", "C"])
        seq_pfunc = functools.partial(extract_backbone_seqs, atoms=["CA"])

        logging.info(
            f"Computing full dataset of {len(fnames)} with {multiprocessing.cpu_count()} threads"
        )
        # Generate dihedral angles
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) #multiprocessing.cpu_count()
        struct_arrays = list(pool.map(pfunc, fnames, chunksize=250))
        coord_arrays = list(pool.map(coords_pfunc, fnames, chunksize=250))
        seq_arrays = list(pool.map(seq_pfunc, fnames, chunksize=250))
        pool.close()
        pool.join()

        # Contains only non-null structures
        structures = []
        for fname, s, c, seq in zip(fnames, struct_arrays, coord_arrays, seq_arrays):
            if s is None:
                continue
            assert len(seq) == len(s)
            structures.append(
                {
                    "angles": s,
                    "coords": c,
                    "seqs": seq,
                    "fname": fname,
                }
            )
        return structures

    def sample_length(self, n: int = 1) -> Union[int, List[int]]:
        """
        Sample a observed length of a sequence
        """
        assert n > 0
        if n == 1:
            l = self._length_rng.choice(self.all_lengths)
        else:
            l = self._length_rng.choice(self.all_lengths, size=n, replace=True).tolist()
        return l

    def get_masked_means(self) -> np.ndarray:
        """Return the means subset to the actual features used"""
        if self.means is None:
            return None
        return self.means

    @functools.cached_property
    def filenames(self) -> List[str]:
        """Return the filenames that constitute this dataset"""
        return [s["fname"] for s in self.structures]

    def __len__(self) -> int:
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(
        self, index, ignore_zero_center: bool = False
    ) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        
        key = self.keys[index]
        if self.db is None:
            self._connect_db()
        structure = pickle.loads(self.db.begin().get(key))

        angles = structure["angles"]
        # NOTE coords are NOT shifted or wrapped, has same length as angles
        coords = structure["coords"].reshape(angles.shape[0],3,3)
        seqs = structure["seqs"]
        seqs = np.array([alphabet_map[one] for one in seqs]).reshape(-1,1)
        assert angles.shape[0] == coords.shape[0]

        if self.means is None:
            self.means = pickle.loads(self.db.begin().get('means'.encode()))
        
        # If given, offset the angles with mean
        if self.means is not None and not ignore_zero_center:
            assert (
                self.means.shape[0] == angles.shape[1]
            ), f"Mismatched shapes: {self.means.shape} != {angles.shape}"
            angles = angles - self.means

            # The distance features all contain a single ":"
            colon_count = np.array([c.count(":") for c in angles.columns])
            # WARNING this uses a very hacky way to find the angles
            angular_idx = np.where(colon_count != 1)[0]
            angles.iloc[:, angular_idx] = modulo_with_wrapped_range(
                angles.iloc[:, angular_idx], -np.pi, np.pi
            )

        # Subset angles to ones we are actaully using as features
        angles = angles.loc[
            :, CathCanonicalAnglesDataset.feature_names["angles"]
        ].values
        assert angles is not None
        assert angles.shape[1] == len(
            CathCanonicalAnglesDataset.feature_is_angular["angles"]
        ), f"Mismatched shapes for angles: {angles.shape[1]} != {len(CathCanonicalAnglesDataset.feature_is_angular['angles'])}"

        # Replace nan values with zero
        np.nan_to_num(angles, copy=False, nan=0)

        # Create attention mask. 0 indicates masked
        l = min(self.pad, angles.shape[0])
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:l] = 1.0

        # Additionally, mask out positions that are nan
        # is_nan = np.where(np.any(np.isnan(angles), axis=1))[0]
        # attn_mask[is_nan] = 0.0  # Mask out the nan positions

        # Perform padding/trimming
        if angles.shape[0] < self.pad:
            angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            coords = np.pad(
                coords,
                ((0, self.pad - coords.shape[0]), (0, 0), (0,0)),
                mode="constant",
                constant_values=0,
            )
            seqs = np.pad(
                seqs,
                ((0, self.pad - seqs.shape[0]), (0, 0)),
                mode="constant",
                constant_values=20,
            )

        elif angles.shape[0] > self.pad:
            if self.trim_strategy == "leftalign":
                angles = angles[: self.pad]
                coords = coords[: self.pad]
                seqs = seqs[:self.pad]
            elif self.trim_strategy == "randomcrop":
                # Randomly crop the sequence to
                start_idx = self.rng.integers(0, angles.shape[0] - self.pad)
                # start_idx = 0
                end_idx = start_idx + self.pad
                assert end_idx < angles.shape[0]
                angles = angles[start_idx:end_idx]
                coords = coords[start_idx:end_idx]
                seqs = seqs[start_idx:end_idx]
                assert angles.shape[0] == coords.shape[0] == self.pad
            else:
                raise ValueError(f"Unknown trim strategy: {self.trim_strategy}")

        # Create position IDs
        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        angular_idx = np.where(CathCanonicalAnglesDataset.feature_is_angular["angles"])[
            0
        ]
        assert tolerant_comparison_check(
            angles[:, angular_idx], ">=", -np.pi
        ), f"Illegal value: {np.min(angles[:, angular_idx])}"
        assert tolerant_comparison_check(
            angles[:, angular_idx], "<=", np.pi
        ), f"Illegal value: {np.max(angles[:, angular_idx])}"
        angles = torch.from_numpy(angles).float()
        coords = torch.from_numpy(coords).float()
        seqs = torch.from_numpy(seqs).long()
        
        # seqs = torch.from_numpy(np.array([alphabet_map[one] for one in seqs])).reshape(-1,1).long()

        retval = {
            "key": key.decode(),
            "seqs": seqs,
            "angles": angles,
            "coords": coords,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
        }
        return retval

    def get_feature_mean_var(self, ft_name: str) -> Tuple[float, float]:
        """
        Return the mean and variance associated with a given feature
        """
        assert ft_name in self.feature_names["angles"], f"Unknown feature {ft_name}"
        idx = self.feature_names["angles"].index(ft_name)
        logging.info(f"Computing metrics for {ft_name} - idx {idx}")

        all_vals = []
        for i in range(len(self)):
            item = self[i]
            attn_idx = torch.where(item["attn_mask"] == 1.0)[0]
            vals = item["angles"][attn_idx, idx]
            all_vals.append(vals)
        all_vals = torch.cat(all_vals)
        assert all_vals.ndim == 1
        return torch.var_mean(all_vals)[::-1]  # Default is (var, mean)
    


class CathCanonicalAnglesOnlyDataset(CathCanonicalAnglesDataset):
    """
    Building on the CATH dataset, return the 3 canonical dihedrals and the 3
    non-dihedral angles. Notably, this does not return distance.
    Dihedrals: phi, psi, omega
    Non-dihedral angles: tau, CA:C:1N, C:1N:1CA
    """

    feature_names = {"angles": ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"], 
                     "coords": ["x", "y", "z"]}
    feature_is_angular = {"angles": [True, True, True, True, True, True],
                          "coords": [False, False, False],}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Trim out the distance in all the feature_names and feature_is_angular
        orig_features = super().feature_names["angles"].copy()
        self.feature_idx = [
            orig_features.index(ft) for ft in self.feature_names["angles"]
        ]
        logging.info(
            f"CATH canonical angles only dataset with {self.feature_names['angles']} (subset idx {self.feature_idx})"
        )

    def get_masked_means(self) -> np.ndarray:
        """Return the means subset to the actual features used"""
        self.means = pickle.loads(self.db.begin().get('means'.encode()))
        return np.copy(self.means)[self.feature_idx]

    def __getitem__(
        self, index, ignore_zero_center: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Return a dict with keys: angles, attn_mask, position_ids
        return_dict = super().__getitem__(index, ignore_zero_center=ignore_zero_center)

        # Remove the distance feature
        assert return_dict["angles"].ndim == 2
        return_dict["angles"] = return_dict["angles"][:, self.feature_idx]
        assert torch.all(
            return_dict["angles"] >= -torch.pi
        ), f"Minimum value {torch.min(return_dict['angles'])} lower than -pi"
        assert torch.all(
            return_dict["angles"] <= torch.pi
        ), f"Maximum value {torch.max(return_dict['angles'])} higher than pi"
        # return_dict.pop("coords", None)

        return return_dict


class CropAnglesDataset(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__ if dset_key
    is not specified; otherwise, returns a dictionary where the item
    to noise is under dset_key

    modulo can be given as either a float or a list of floats
    """

    def __init__(
        self,
        dset: Dataset,
        dset_key: str = "angles",
        timesteps: int = 250,
        exhaustive_t: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "cosine",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
    ) -> None:
        super().__init__()
        self.dset = dset
        assert hasattr(dset, "feature_names")
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        assert (
            dset_key in dset.feature_is_angular
        ), f"{dset_key} not in {dset.feature_is_angular}"
        self.n_features = len(dset.feature_is_angular[dset_key])

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)
        self.rng = np.random.default_rng(seed=6489)

    @property
    def feature_names(self):
        """Pass through feature names property of wrapped dset"""
        return self.dset.feature_names

    @property
    def feature_is_angular(self):
        """Pass through feature is angular property of wrapped dset"""
        return self.dset.feature_is_angular

    @property
    def pad(self):
        """Pas through the pad property of wrapped dset"""
        return self.dset.pad

    @property
    def filenames(self):
        """Pass through the filenames property of the wrapped dset"""
        return self.dset.filenames

    def sample_length(self, *args, **kwargs):
        return self.dset.sample_length(*args, **kwargs)

    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with {self.schedule}-{self.timesteps} with variance scales {self.nonangular_var_scale} and {self.angular_var_scale}"

    def __len__(self) -> int:
        if not self.exhaustive_timesteps:
            return len(self.dset)
        else:
            return int(len(self.dset) * self.timesteps)

    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Gets the i-th item in the dataset and adds noise
        use_t_val is useful for manually querying specific timepoints
        """
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        # Handle cases where we exhaustively loop over t
        item = self.dset.__getitem__(index, ignore_zero_center=ignore_zero_center)

        
        L = item['attn_mask'].sum().long()
        start_idx = self.rng.integers(0, L-6)
        end_idx = self.rng.integers(start_idx+5, L)

        attn_mask = item['attn_mask'][start_idx:end_idx]
        angles = item['angles'][start_idx:end_idx]
        coords = item['coords'][start_idx:end_idx]
        
        attn_mask = np.pad(
                attn_mask,
                ((0, self.pad - attn_mask.shape[0])),
                mode="constant",
                constant_values=0,
            )
        angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        coords = np.pad(
            coords,
            ((0, self.pad - coords.shape[0]), (0, 0), (0,0)),
            mode="constant",
            constant_values=0,
        )
        
        item['attn_mask'] = attn_mask
        item['angles'] = angles
        item['coords'] = coords
        
        return item

class NoisedAnglesDataset(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__ if dset_key
    is not specified; otherwise, returns a dictionary where the item
    to noise is under dset_key

    modulo can be given as either a float or a list of floats
    """

    def __init__(
        self,
        dset: Dataset,
        dset_key: str = "angles",
        timesteps: int = 250,
        exhaustive_t: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "cosine",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
    ) -> None:
        super().__init__()
        self.dset = dset
        assert hasattr(dset, "feature_names")
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        assert (
            dset_key in dset.feature_is_angular
        ), f"{dset_key} not in {dset.feature_is_angular}"
        self.n_features = len(dset.feature_is_angular[dset_key])

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)
        self.rng = np.random.default_rng(seed=6489)

    @property
    def feature_names(self):
        """Pass through feature names property of wrapped dset"""
        return self.dset.feature_names

    @property
    def feature_is_angular(self):
        """Pass through feature is angular property of wrapped dset"""
        return self.dset.feature_is_angular

    @property
    def pad(self):
        """Pas through the pad property of wrapped dset"""
        return self.dset.pad

    @property
    def filenames(self):
        """Pass through the filenames property of the wrapped dset"""
        return self.dset.filenames

    def sample_length(self, *args, **kwargs):
        return self.dset.sample_length(*args, **kwargs)

    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with {self.schedule}-{self.timesteps} with variance scales {self.nonangular_var_scale} and {self.angular_var_scale}"

    def __len__(self) -> int:
        if not self.exhaustive_timesteps:
            return len(self.dset)
        else:
            return int(len(self.dset) * self.timesteps)

    def plot_alpha_bar_t(self, fname: str) -> str:
        """Plot the alpha bar for each timestep"""
        fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
        vals = self.alphas_cumprod.numpy()
        ax.plot(np.arange(len(vals)), vals)
        ax.set(
            ylabel=r"$\bar \alpha_t$",
            xlabel=r"Timestep $t$",
            title=f"Alpha bar for {self.schedule} across {self.timesteps} timesteps",
        )
        fig.savefig(fname, bbox_inches="tight")
        return fname

    def sample_noise(self, vals: torch.Tensor) -> torch.Tensor:
        """
        Adaptively sample noise based on modulo. We scale only the variance because
        we want the noise to remain zero centered
        """
        # Noise is always 0 centered
        noise = torch.randn_like(vals)

        # Shapes of vals couled be (batch, seq, feat) or (seq, feat)
        # Therefore we need to index into last dimension consistently

        # Scale by provided variance scales based on angular or not
        if self.angular_var_scale != 1.0 or self.nonangular_var_scale != 1.0:
            for j in range(noise.shape[-1]):  # Last dim = feature dim
                s = (
                    self.angular_var_scale
                    if self.dset.feature_is_angular[self.dset_key][j]
                    else self.nonangular_var_scale
                )
                noise[..., j] *= s

        # Make sure that the noise doesn't run over the boundaries
        angular_idx = np.where(self.dset.feature_is_angular[self.dset_key])[0]
        noise[..., angular_idx] = modulo_with_wrapped_range(
            noise[..., angular_idx], -np.pi, np.pi
        )

        return noise

    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Gets the i-th item in the dataset and adds noise
        use_t_val is useful for manually querying specific timepoints
        """
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        # Handle cases where we exhaustively loop over t
        if self.exhaustive_timesteps:
            item_index = index // self.timesteps
            assert item_index < len(self.dset)
            time_index = index % self.timesteps
            logging.debug(
                f"Exhaustive {index} -> item {item_index} at time {time_index}"
            )
            assert (
                item_index * self.timesteps + time_index == index
            ), f"Unexpected indices for {index} -- {item_index} {time_index}"
            item = self.dset.__getitem__(
                item_index, ignore_zero_center=ignore_zero_center
            )
        else:
            item = self.dset.__getitem__(index, ignore_zero_center=ignore_zero_center)

        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key].clone()
        else:
            vals = item.clone()
        assert isinstance(
            vals, torch.Tensor
        ), f"Using dset_key {self.dset_key} - expected tensor but got {type(vals)}"

        # Sample a random timepoint and add corresponding noise
        if use_t_val is not None:
            assert (
                not self.exhaustive_timesteps
            ), "Cannot use specific t in exhaustive mode"
            t_val = np.clip(np.array([use_t_val]), 0, self.timesteps - 1)
            t = torch.from_numpy(t_val).long()
        elif self.exhaustive_timesteps:
            t = torch.tensor([time_index]).long()  # list to get correct shape
        else:
            t = torch.randint(0, self.timesteps, (1,)).long()

        # Get the values for alpha and beta
        sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][t.item()]
        sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms[
            "sqrt_one_minus_alphas_cumprod"
        ][t.item()]
        # Noise is sampled within range of [-pi, pi], and optionally
        # shifted to [0, 2pi] by adding pi
        noise = self.sample_noise(vals)  # Vals passed in only for shape

        # Add noise and ensure noised vals are still in range
        noised_vals = (
            sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
        )
        assert noised_vals.shape == vals.shape, f"Unexpected shape {noised_vals.shape}"
        # The underlying vals are already shifted, and noise is already shifted
        # All we need to do is ensure we stay on the corresponding manifold
        angular_idx = np.where(self.dset.feature_is_angular[self.dset_key])[0]
        # Wrap around the correct range
        noised_vals[:, angular_idx] = modulo_with_wrapped_range(
            noised_vals[:, angular_idx], -np.pi, np.pi
        )

        L = item['attn_mask'].sum().long()
        unknown_mask = torch.zeros(vals.shape[0],1)==1
        mask_length = self.rng.integers(5, L/3)
        start_idx = self.rng.integers(0, L - mask_length-1)
        end_idx = start_idx + mask_length
        unknown_mask[start_idx+1:end_idx] = True

        retval = {
            "unknown_mask": unknown_mask,
            "corrupted": noised_vals,
            "t": t,
            "known_noise": noise,
            "start_idx": start_idx,
            "end_idx": end_idx
        }

        # Update dictionary if wrapped dset returns dicts, else just return
        if isinstance(item, dict):
            assert item.keys().isdisjoint(retval.keys())
            item.update(retval)
            return item
        return retval


class NoisedAnglesDataset_General(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__ if dset_key
    is not specified; otherwise, returns a dictionary where the item
    to noise is under dset_key

    modulo can be given as either a float or a list of floats
    """

    def __init__(
        self,
        dset: Dataset,
        dset_key: str = "angles",
        timesteps: int = 250,
        exhaustive_t: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "cosine",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
        ignore_zero_center = False,
        sampling = False
    ) -> None:
        super().__init__()
        self.sampling = sampling
        self.dset = dset
        self.ignore_zero_center = ignore_zero_center
        assert hasattr(dset, "feature_names")
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        assert (
            dset_key in dset.feature_is_angular
        ), f"{dset_key} not in {dset.feature_is_angular}"
        self.n_features = len(dset.feature_is_angular[dset_key])

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)
        self.rng = np.random.default_rng(seed=6489)
        self.data_builder = TorchNERFBuilder(0)

    @property
    def feature_names(self):
        """Pass through feature names property of wrapped dset"""
        return self.dset.feature_names

    @property
    def feature_is_angular(self):
        """Pass through feature is angular property of wrapped dset"""
        return self.dset.feature_is_angular

    @property
    def pad(self):
        """Pas through the pad property of wrapped dset"""
        return self.dset.pad

    @property
    def filenames(self):
        """Pass through the filenames property of the wrapped dset"""
        return self.dset.filenames

    def sample_length(self, *args, **kwargs):
        return self.dset.sample_length(*args, **kwargs)

    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with {self.schedule}-{self.timesteps} with variance scales {self.nonangular_var_scale} and {self.angular_var_scale}"

    def __len__(self) -> int:
        if not self.exhaustive_timesteps:
            return len(self.dset)
        else:
            return int(len(self.dset) * self.timesteps)

    def plot_alpha_bar_t(self, fname: str) -> str:
        """Plot the alpha bar for each timestep"""
        fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
        vals = self.alphas_cumprod.numpy()
        ax.plot(np.arange(len(vals)), vals)
        ax.set(
            ylabel=r"$\bar \alpha_t$",
            xlabel=r"Timestep $t$",
            title=f"Alpha bar for {self.schedule} across {self.timesteps} timesteps",
        )
        fig.savefig(fname, bbox_inches="tight")
        return fname

    

    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Gets the i-th item in the dataset and adds noise
        use_t_val is useful for manually querying specific timepoints
        """
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        
        ignore_zero_center = self.ignore_zero_center
        # Handle cases where we exhaustively loop over t
        if self.exhaustive_timesteps:
            item_index = index // self.timesteps
            assert item_index < len(self.dset)
            time_index = index % self.timesteps
            logging.debug(
                f"Exhaustive {index} -> item {item_index} at time {time_index}"
            )
            assert (
                item_index * self.timesteps + time_index == index
            ), f"Unexpected indices for {index} -- {item_index} {time_index}"
            item = self.dset.__getitem__(
                item_index, ignore_zero_center=ignore_zero_center
            )
        else:
            item = self.dset.__getitem__(index, ignore_zero_center=ignore_zero_center)

        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key].clone()
        else:
            vals = item.clone()
        assert isinstance(
            vals, torch.Tensor
        ), f"Using dset_key {self.dset_key} - expected tensor but got {type(vals)}"

        # Sample a random timepoint and add corresponding noise
        if use_t_val is not None:
            assert (
                not self.exhaustive_timesteps
            ), "Cannot use specific t in exhaustive mode"
            t_val = np.clip(np.array([use_t_val]), 0, self.timesteps - 1)
            t = torch.from_numpy(t_val).long()
        elif self.exhaustive_timesteps:
            t = torch.tensor([time_index]).long()  # list to get correct shape
        else:
            t = torch.randint(0, self.timesteps, (1,)).long()


        L = item['attn_mask'].sum().long()
        unknown_mask = torch.zeros(vals.shape[0],1)==1
        mask_length = self.rng.integers(5, L/3)
        start_idx = self.rng.integers(0, L - mask_length-1)
        end_idx = start_idx + mask_length
        unknown_mask[start_idx+1:end_idx] = True
        
        
        if self.sampling:
            ## only for sampling_foldingdiff, match duaspace
            mask_location = json.load(open("/gaozhangyang/experiments/DiffSDS/model_zoom/mask_location.json", "r"))
            
            start_idx, end_idx = mask_location[item['key']]
            
            unknown_mask = torch.zeros(vals.shape[0],1)==1
            unknown_mask[start_idx+1:end_idx] = True
            # -----------------------------------
            

        

        retval = {
            "unknown_mask": unknown_mask,
            "t": t,
            "start_idx": start_idx,
            "end_idx": end_idx
        }

        # Update dictionary if wrapped dset returns dicts, else just return
        if isinstance(item, dict):
            assert item.keys().isdisjoint(retval.keys())
            item.update(retval)
            return item
        return retval


class DataLoader_Protein(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,collate_fn=None, **kwargs):
        super(DataLoader_Protein, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)

def get_dataset(config, distributed=True, sampling=False):
    from .load_data import DataLoader_Protein, CathCanonicalAnglesOnlyDataset, NoisedAnglesDataset, CropAnglesDataset
    
    
    clean_train_dataset = CathCanonicalAnglesOnlyDataset(split = "train", pad = config["pad"], min_length = config["min_length"], trim_strategy='randomcrop', zero_center=True,)

    clean_valid_dataset = CathCanonicalAnglesOnlyDataset(split = "valid", pad = config["pad"], min_length = config["min_length"], trim_strategy='randomcrop', zero_center=True,)

    clean_test_dataset = CathCanonicalAnglesOnlyDataset(split = "test", pad = config["pad"], min_length = config["min_length"], trim_strategy='randomcrop', zero_center=True, strict_test=config['strict_test'])


    ### for FoldingDiff
    # train_dataset = NoisedAnglesDataset(clean_train_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)

    # valid_dataset = NoisedAnglesDataset(clean_valid_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)

    # test_dataset = NoisedAnglesDataset(clean_test_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)
    
    #### for PredEnd
    # train_dataset = CropAnglesDataset(clean_train_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)

    # valid_dataset = CropAnglesDataset(clean_valid_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)

    # test_dataset = CropAnglesDataset(clean_test_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0)
    
    
    # ## for CFoldingDiff
    if config['method'] in ["CFoldingDiff", "FoldingDiff"]:
        train_dataset = NoisedAnglesDataset_General(clean_train_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, sampling=sampling)

        valid_dataset = NoisedAnglesDataset_General(clean_valid_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, sampling=sampling)

        test_dataset = NoisedAnglesDataset_General(clean_test_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, sampling=sampling)
    
    
    ## for DualSpace
    if config['method'] in ["DiffSDS"]:
        train_dataset = NoisedAnglesDataset_General(clean_train_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, ignore_zero_center=config['ignore_zero_center'])

        valid_dataset = NoisedAnglesDataset_General(clean_valid_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, ignore_zero_center=config['ignore_zero_center'])

        test_dataset = NoisedAnglesDataset_General(clean_test_dataset, dset_key="angles", timesteps=config["timesteps"], exhaustive_t=False, beta_schedule="cosine", nonangular_variance=1.0, angular_variance=1.0, ignore_zero_center=config['ignore_zero_center'], sampling=sampling)
    
    


    if distributed:
        dataset_sample_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader_Protein(train_dataset, int(config['batch_size']/config["nproc_per_node"]), sampler=dataset_sample_train, num_workers=config['num_workers'])

        dataset_sample_valid = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader_Protein(valid_dataset, int(config['batch_size']/config["nproc_per_node"]), sampler=dataset_sample_valid, num_workers=config['num_workers'])

        dataset_sample_test = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = DataLoader_Protein(test_dataset, int(config['batch_size']/config["nproc_per_node"]), sampler=dataset_sample_test, num_workers=config['num_workers'])
    else:

        train_loader = DataLoader_Protein(train_dataset, config['batch_size'], shuffle=True, num_workers=config['num_workers'])

        valid_loader = DataLoader_Protein(valid_dataset, config['batch_size'], num_workers=config['num_workers'])

        test_loader = DataLoader_Protein(test_dataset, config['batch_size'], num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    dataset = CathCanonicalAnglesOnlyDataset(split = "train", pad = 128, min_length=40, trim_strategy='randomcrop', zero_center=True, toy=None)
    dataset[0]
    print()