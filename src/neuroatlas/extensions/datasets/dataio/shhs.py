"""
SHHS (Sleep Heart Health Study) dataset loader for EEGBenchmarks.
Adapted from ~/Documents/CoRe-Sleep/datasets/sleepset.py.

Provides SleepDataLoader with .train_loader / .valid_loader / .test_loader
yielding batches: {"data": {"stft_eeg": ..., "stft_eog": ...}, "label": ..., ...}
"""

__all__ = ["SleepDataLoader"]

from pathlib import Path

import einops
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import torch
import os
import copy
import random
import pickle
import h5py
from tqdm import tqdm
from scipy.io import loadmat
from collections import defaultdict
import logging
import easydict
from os.path import join as pjoin
from neuroatlas.benchmarking_helpers import dataloader_worker_init_fn

# Default path for bundled assets shipped with this module
_ASSETS_DIR = Path(__file__).parent / "assets"

nested_dict = lambda: defaultdict(nested_dict)


class Sleep_Dataset(Dataset):

    def __init__(self, config: easydict.EasyDict, views: dict, set_name: str):
        super()
        self.dataset = views
        self.views = list(self.dataset.keys())
        self.config = config
        self.set_name = set_name

        self._init_attributes()

        if self.filter_patients["use_type"]:
            self._find_list_of_patients()
            self.broken_mod_dict = self._get_broken_modalities(
                filename=self.config.dataset.broken_patients_filepath)

        self._get_cumulatives()

    def _init_attributes(self):
        self.num_views = len(self.dataset)
        self.normalize = True
        self.outer_seq_length = self.config.dataset.outer_seq_length
        self.filter_patients = self.config.dataset.filter_patients[self.set_name]

    def _get_cumulatives(self):
        view = list(self.dataset.keys())[0]
        self.cumulatives = {"lengths": [0], "files": {}}
        for file_idx in range(len(self.dataset[view]["dataset"])):
            file = self.dataset[view]["dataset"][file_idx]
            patient_num = int(file["filename"].split("/")[-1][1:5])
            file_len = int(file["len_windows"])

            if self.filter_patients["use_type"] == "include_only_skipped":
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            elif self.filter_patients["use_type"] == "subsample":
                if not self.filter_patients["whole_patient"]:
                    raise Warning("Whole patients is not true.")
                self._single_patient_cumulative_includeskipped(patient_num, file_len, file_idx)
            else:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def _find_list_of_patients(self):
        self.patient_list = []
        for view in self.dataset:
            for file_idx in range(len(self.dataset[view]["dataset"])):
                file = self.dataset[view]["dataset"][file_idx]
                patient_num = int(file["filename"].split("/")[-1][1:5])
                self.patient_list.append(patient_num)
        self.patient_list = np.unique(np.array(self.patient_list))

    def _load_n_norm_mat(self, file_info: dict, patient_idx: int, mod: str) -> dict:
        file_name = file_info["dataset"][mod]["filename"]
        data_idx = file_info["data_pos"]
        data_num = file_info["data_num"]
        end_file = file_info["end_file"]
        start_file = file_info["start_file"]
        end_idx = data_num + data_idx

        if "skip_skips" not in self.filter_patients or self.filter_patients["skip_skips"]:
            skip_view = torch.empty(0)
        else:
            skip_view = file_info["skip_views"][mod]

        f = h5py.File(file_name, 'r', swmr=True)
        if "stft" in mod:
            signal = f["X2"][:, :, data_idx:end_idx]
            signal = np.expand_dims(signal, axis=1)
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                signal = einops.rearrange(signal,
                                          "freq channels time inner -> inner time channels freq")
                signal = (signal - self.mean[mod]["ch_0"]) / self.std[mod]["ch_0"]
                signal = einops.rearrange(signal,
                                          "inner time channels freq -> inner channels freq time")
            img = torch.from_numpy(signal).unsqueeze(dim=2)
        else:
            # raw timeseries — X1 is (n_samples, n_epochs), e.g. (3000, N) at 100 Hz
            signal = f["X1"][:, data_idx:end_idx].T.copy()  # (N, 3000)
            signal = np.expand_dims(signal, axis=1).astype(np.float32)  # (N, 1, 3000)
            if self.normalize and hasattr(self, "mean") and hasattr(self, "std"):
                mean_val = self.mean.get(mod, 0.0)
                std_val = self.std.get(mod, 1.0)
                if isinstance(mean_val, dict):
                    mean_val = mean_val.get("ch_0", 0.0)
                    std_val = std_val.get("ch_0", 1.0)
                signal = (signal - mean_val) / (float(std_val) + 1e-8)
            img = torch.from_numpy(signal)  # (N, 1, 3000) — no extra unsqueeze

        label = f["label"][0, data_idx:end_idx]
        init = torch.zeros(len(label))
        if data_idx == start_file and end_file > data_idx and len(init) > 0:
            init[0] = 1
        elif end_idx == end_file:
            init[-1] = 1
        label = torch.from_numpy(label).long() - 1

        ids = [{"patient_num": patient_idx, "ids": i} for i in range(data_idx, end_idx)]

        return {"data": img, "label": label, "init": init, "skip_view": skip_view, "ids": ids}

    def _find_file_to_open(self, file_cumul_idx, previous_output) -> dict:
        if "index" not in previous_output:
            raise ValueError("Missing attribute 'index' in previous_output")
        if "remaining_sleep_epochs" not in previous_output:
            raise ValueError("Missing attribute 'remaining_sleep_epochs' in previous_output")

        index = previous_output["index"]
        remaining_sleep_epochs = previous_output["remaining_sleep_epochs"]

        patient_num = -1
        lo = self.cumulatives["lengths"][file_cumul_idx]
        hi = self.cumulatives["lengths"][file_cumul_idx + 1]
        if hi > index and lo <= index:
            cumul_file = self.cumulatives["files"]["{}-{}".format(lo, hi)]
            data_idx = index - lo + cumul_file["data_idx"]["start_idx"]
            new_remaining = max(remaining_sleep_epochs - (cumul_file["data_idx"]["end_idx"] - data_idx), 0)
            data_num = remaining_sleep_epochs - new_remaining

            previous_output["index"] += data_num
            previous_output["remaining_sleep_epochs"] = new_remaining
            new_output = {
                "data_pos": data_idx,
                "data_num": data_num,
                "end_file": cumul_file["data_idx"]["end_idx"],
                "start_file": cumul_file["data_idx"]["start_idx"],
                "remaining_sleep_epochs": previous_output["remaining_sleep_epochs"],
                "dataset": cumul_file["dataset"],
            }
            patient_num = cumul_file["patient_num"]

            if "skip_views" not in cumul_file:
                new_output["skip_views"] = {view: torch.empty(0) for view in self.dataset}
            elif (type(cumul_file["skip_views"][list(cumul_file["skip_views"].keys())[0]]) == torch.Tensor
                  and ("skip_skips" in self.filter_patients
                       and not self.filter_patients["skip_skips"])):
                new_output["skip_views"] = {
                    view: cumul_file["skip_views"][view][
                        data_idx - cumul_file["data_idx"]["start_idx"]:
                        data_idx + data_num - cumul_file["data_idx"]["start_idx"]]
                    for view in self.dataset
                }

            if cumul_file["patient_num"] in previous_output:
                previous_output[cumul_file["patient_num"]].append(new_output)
            else:
                previous_output.update({cumul_file["patient_num"]: [new_output]})

        return previous_output, patient_num

    def choose_specific_patient(self, patient_nums, include_chosen=True):
        for view in self.dataset:
            new_view_dataset = []
            for file in self.dataset[view]["dataset"]:
                if include_chosen:
                    if int(file["filename"].split("/")[-1][1:5]) in patient_nums:
                        new_view_dataset.append(file)
                else:
                    if int(file["filename"].split("/")[-1][1:5]) not in patient_nums:
                        new_view_dataset.append(file)
            self.dataset[view]["dataset"] = new_view_dataset
        self._get_cumulatives()

    def _subsample_patients(self, std_per_indices):
        if "subsets" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'subsets' key.")
        required_modalities = ["combined", "eeg", "eog"]
        for modality in required_modalities:
            if modality not in self.filter_patients["subsets"]:
                raise ValueError(f"'subsets' must contain '{modality}' key.")

        combined_set = random.sample(list(self.patient_list),
                                     self.filter_patients["subsets"]["combined"])
        skip_patient_ids = {
            patient: torch.cat([
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
            ], dim=1)
            for patient in combined_set
        }
        not_chosen_patients = [p for p in self.patient_list if p not in combined_set]

        eeg_set = random.sample(not_chosen_patients, self.filter_patients["subsets"]["eeg"])
        skips = {
            patient: torch.cat([
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
            ], dim=1)
            for patient in eeg_set
        }
        skip_patient_ids.update(skips)
        not_chosen_patients = [p for p in self.patient_list if p not in combined_set]

        eog_set = random.sample(not_chosen_patients, self.filter_patients["subsets"]["eog"])
        skips = {
            patient: torch.cat([
                torch.ones(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
                torch.zeros(len(std_per_indices[patient]["std_eeg"])).unsqueeze(dim=1),
            ], dim=1)
            for patient in eog_set
        }
        skip_patient_ids.update(skips)

        self.subsampled_patients = {
            "combined": combined_set, "eeg": eeg_set, "eog": eog_set,
            "all": combined_set + eeg_set + eog_set,
        }
        logging.debug("Subsample: {} both, {} EEG-only, {} EOG-only.".format(
            len(combined_set), len(eeg_set), len(eog_set)))
        return skip_patient_ids

    def _get_broken_modalities(self, filename):
        with open(filename, "rb") as file:
            std_per_indices = pickle.load(file)

        if self.filter_patients["use_type"] == "subsample":
            return self._subsample_patients(std_per_indices)

        if "std_threshold" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'std_threshold' key.")
        if "perc_threshold" not in self.filter_patients:
            raise ValueError("'filter_patients' must contain a 'perc_threshold' key.")

        threshold = self.filter_patients["std_threshold"]
        perc_threshold = self.filter_patients["perc_threshold"]

        mod_diff = {
            i: (std_per_indices[i]["std_eeg"][:, 2] - std_per_indices[i]["std_eog"][:, 2]).numpy()
            for i in std_per_indices.keys() if i in self.patient_list
        }

        perc_t = {i: (np.abs(mod_diff[i]) > threshold).sum() / len(mod_diff[i]) for i in mod_diff}
        patients_chosen = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        logging.debug("Patients with broken modalities: {}".format(len(patients_chosen)))

        perc_t = {i: (mod_diff[i] > threshold).sum() / len(mod_diff[i]) for i in mod_diff}
        patients_chosen_eeg = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        logging.debug("Patients with broken EEG: {}".format(len(patients_chosen_eeg)))

        perc_t = {i: (-mod_diff[i] > threshold).sum() / len(mod_diff[i]) for i in mod_diff}
        patients_chosen_eog = np.array([i for i in perc_t if perc_t[i] > perc_threshold])
        logging.debug("Patients with broken EOG: {}".format(len(patients_chosen_eog)))

        skip_patient_ids = {}
        for i in patients_chosen:
            skip_mod = torch.zeros(len(mod_diff[i]), 2)
            skip_mod[mod_diff[i] > threshold, 0] = 1
            skip_mod[mod_diff[i] < -threshold, 1] = 1
            skip_patient_ids[i] = skip_mod
        return skip_patient_ids

    def _single_patient_cumulative_includeskipped(self, patient_num, file_len, file_idx):
        if patient_num in self.broken_mod_dict:
            if self.filter_patients["whole_patient"]:
                self._single_patient_cumulative_full(patient_num, file_len, file_idx)
                return

            this_broken = copy.deepcopy(self.broken_mod_dict[patient_num])
            this_broken[:, 1:2] *= 2
            skip_labels, skip_labels_lengths = torch.unique_consecutive(
                this_broken.sum(dim=1), return_counts=True)
            count, consecutives = 0, []
            for i in range(len(skip_labels)):
                end = count + skip_labels_lengths[i]
                consecutives.append({
                    "start": count, "end": end, "skip_label": skip_labels[i],
                    "skip_views": {
                        "stft_eeg": self.broken_mod_dict[patient_num][
                            count:count + skip_labels_lengths[i], :1].squeeze(),
                        "stft_eog": self.broken_mod_dict[patient_num][
                            count:count + skip_labels_lengths[i], 1:].squeeze(),
                    }
                })
                count = count + skip_labels_lengths[i]

            for cons in consecutives:
                if cons["skip_label"] == 3 or cons["skip_label"] == 0:
                    continue
                self.cumulatives["lengths"].append(
                    cons["end"] - cons["start"] + self.cumulatives["lengths"][-1])
                self.cumulatives["files"]["{}-{}".format(
                    self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
                    "patient_num": patient_num,
                    "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                    "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset},
                    "skip_views": cons["skip_views"],
                }

    def _single_patient_cumulative_subsample(self, patient_num, file_len, file_idx):
        self.cumulatives["lengths"].append(file_len + self.cumulatives["lengths"][-1])
        self.cumulatives["files"]["{}-{}".format(
            self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
            "patient_num": patient_num,
            "data_idx": {"start_idx": 0, "end_idx": file_len},
            "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset},
        }

    def _single_patient_cumulative_full(self, patient_num, file_len, file_idx):
        start_idx, end_idx = 0, file_len

        if hasattr(self, "broken_mod_dict") and patient_num in self.broken_mod_dict:
            this_broken = copy.deepcopy(self.broken_mod_dict[patient_num])
            this_broken[:, 1:2] *= 2
            skip_labels, skip_labels_lengths = torch.unique_consecutive(
                this_broken.sum(dim=1), return_counts=True)
            count, consecutives = 0, []
            for i in range(len(skip_labels)):
                end = count + skip_labels_lengths[i]
                consecutives.append({
                    "start": count, "end": end, "skip_label": skip_labels[i],
                    "skip_views": {
                        "stft_eeg": self.broken_mod_dict[patient_num][
                            count:count + skip_labels_lengths[i], :1].squeeze(),
                        "stft_eog": self.broken_mod_dict[patient_num][
                            count:count + skip_labels_lengths[i], 1:].squeeze(),
                    }
                })
                count = count + skip_labels_lengths[i]
        else:
            consecutives = [{
                "start": start_idx, "end": end_idx,
                "skip_views": {
                    "stft_eeg": torch.zeros(end_idx - start_idx),
                    "stft_eog": torch.zeros(end_idx - start_idx),
                }
            }]

        for cons in consecutives:
            self.cumulatives["lengths"].append(
                cons["end"] - cons["start"] + self.cumulatives["lengths"][-1])
            self.cumulatives["files"]["{}-{}".format(
                self.cumulatives["lengths"][-2], self.cumulatives["lengths"][-1])] = {
                "patient_num": patient_num,
                "data_idx": {"start_idx": cons["start"], "end_idx": cons["end"]},
                "dataset": {view: self.dataset[view]["dataset"][file_idx] for view in self.dataset},
                "skip_views": cons["skip_views"],
            }

    def _load_view_files(self, file_output, view):
        view_output = defaultdict(lambda: [])
        for patient_idx, this_patient_instr in file_output.items():
            for seqs in this_patient_instr:
                loaded = self._load_n_norm_mat(file_info=seqs, patient_idx=patient_idx, mod=view)
                for i in loaded:
                    view_output[i].append(loaded[i])
        return dict(view_output)

    def _aggegate_n_update_view(self, output, view_output, view):
        for out_i in view_output:
            if out_i == "ids":
                view_output["ids"] = [item for sublist in view_output["ids"] for item in sublist]
                ids = torch.cat([torch.Tensor([int(id["ids"])]) for id in view_output["ids"]])
                patient_nums = torch.cat(
                    [torch.Tensor([int(id["patient_num"])]) for id in view_output["ids"]])
                total_ids = torch.cat([patient_nums.unsqueeze(dim=1), ids.unsqueeze(dim=1)], dim=1)
                output[out_i].update({view: total_ids})
            else:
                output[out_i].update({view: torch.cat(view_output[out_i])})

    def __getitem__(self, index):
        index = index * self.outer_seq_length

        file_output = {"remaining_sleep_epochs": self.outer_seq_length, "index": index}
        for file_cumul_idx in range(len(self.cumulatives["lengths"]) - 1):
            file_output, last_pat_num = self._find_file_to_open(
                file_cumul_idx=file_cumul_idx, previous_output=file_output)
            if ((last_pat_num in file_output)
                    and file_output[last_pat_num][-1]["remaining_sleep_epochs"] == 0):
                break

        file_output.pop("index")
        file_output.pop("remaining_sleep_epochs")

        output = nested_dict()
        for view in self.views:
            view_output = self._load_view_files(file_output=file_output, view=view)
            self._aggegate_n_update_view(output=output, view_output=view_output, view=view)

        primary_view = self.views[0]
        output["idx"] = output["ids"][primary_view]
        output["label"] = output["label"][primary_view]
        return output

    def __len__(self):
        return int(self.cumulatives["lengths"][-1] / self.outer_seq_length)


class SleepDataLoader:

    def __init__(self, config: easydict.EasyDict):
        self.config = config

        sleep_dataset_train, sleep_dataset_val, sleep_dataset_test, sleep_dataset_total = (
            self._get_datasets())

        num_cores = max(len(os.sched_getaffinity(0)) - 1, 0)
        explicit_workers = self.config.training_params.data_loader_workers
        train_workers = explicit_workers if explicit_workers == 0 else num_cores
        logging.debug("Available cores: {}, using: {}".format(len(os.sched_getaffinity(0)), train_workers))

        self.train_loader = torch.utils.data.DataLoader(
            sleep_dataset_train,
            batch_size=self.config.training_params.batch_size,
            num_workers=train_workers,
            pin_memory=self.config.training_params.pin_memory,
            worker_init_fn=dataloader_worker_init_fn,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            sleep_dataset_val,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=self.config.training_params.data_loader_workers,
            pin_memory=self.config.training_params.pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            sleep_dataset_test,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=self.config.training_params.data_loader_workers,
            pin_memory=self.config.training_params.pin_memory,
        )
        self.total_loader = torch.utils.data.DataLoader(
            sleep_dataset_total,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=self.config.training_params.data_loader_workers,
            pin_memory=self.config.training_params.pin_memory,
        )

        self.norm_agent = Normalization_finder(dataloader=self, config=config)
        self.metrics = self.norm_agent.get_norm_metrics()

        logging.info("Train: {}, Val: {}, Test: {}".format(
            len(self.train_loader), len(self.valid_loader), len(self.test_loader)))

        if self.config.get("statistics", {}).get("print", False):
            self._statistics_mat()

    def _get_datasets(self):
        views = {}
        for i in self.config.dataset.data_view_dir:
            i = dict(i)  # ensure mutable copy
            i["list_dir"] = pjoin(self.config.dataset.data_roots, i["list_dir"])
            views[i["data_type"] + "_" + i["mod"]] = i
        views = self._read_dirs_mat(views)
        train_views, val_views, test_views = self._split_data_mat(views)

        train_dataset  = Sleep_Dataset(config=self.config, views=train_views,  set_name="train")
        valid_dataset  = Sleep_Dataset(config=self.config, views=val_views,    set_name="val")
        test_dataset   = Sleep_Dataset(config=self.config, views=test_views,   set_name="test")
        total_dataset  = Sleep_Dataset(config=self.config, views=views,        set_name="total")
        return train_dataset, valid_dataset, test_dataset, total_dataset

    def _read_dirs_mat(self, view_dirs):
        for view in view_dirs:
            list_dir = view_dirs[view]["list_dir"]
            dataset = []
            if not os.path.isfile(list_dir):
                # SHHS is read from the authors' preprocessed copy (as `data
                # status shhs` says), whose recording list this is
                raise FileNotFoundError(
                    f"shhs: the preprocessed SHHS copy is not in {os.path.dirname(list_dir)} "
                    f"(no {os.path.basename(list_dir)} there)\n"
                    f"fix: ask the authors for the preprocessed copy, then neuroatlas config "
                    f"set shhs.data_root DIR")
            with open(list_dir) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                for j, row in enumerate(csv_reader):
                    dt = row[0].split("-")
                    dataset.append({"filename": dt[0], "len_windows": dt[1]})
            view_dirs[view]["dataset"] = dataset
        return view_dirs

    def _split_data_mat(self, views):
        split_method = self.config.dataset.data_split.split_method
        if split_method == "patients_test":
            return self._split_patients_test(
                dirs_train_whole=views,
                split_rate_val=self.config.dataset.data_split.val_split_rate,
                split_rate_test=self.config.dataset.data_split.test_split_rate,
            )
        elif split_method == "patients_sleeptransformer":
            logging.info("Splitting dataset by SleepTransformer split")
            return self._split_patients_sleeptf(dirs_train_whole=views)
        else:
            raise ValueError("Unknown split method: {}".format(split_method))

    def _split_patients_test(self, dirs_train_whole, split_rate_val, split_rate_test):
        splits_file = self.config.dataset.data_split.get(
            "trainvaltest_splits_file",
            str(_ASSETS_DIR / "trainvaltest_splits.pkl"))
        with open(splits_file, "rb") as f:
            splits = pickle.load(f)

        train_views = copy.deepcopy(dirs_train_whole)
        val_views   = copy.deepcopy(dirs_train_whole)
        test_views  = copy.deepcopy(dirs_train_whole)
        this_split  = splits[self.config.dataset.fold]

        for view in dirs_train_whole:
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = file["filename"].split("/")[-1][:5]
                if patient_num in this_split["test"]:
                    test_dataset.append(file)
                elif patient_num in this_split["train"]:
                    train_dataset.append(file)
                elif patient_num in this_split["val"]:
                    val_dataset.append(file)
                else:
                    raise Warning("Patient {} has no split assignment.".format(file))
            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"]   = val_dataset
            test_views[view]["dataset"]  = test_dataset
            logging.info("{}: train={}, val={}, test={}".format(
                view, len(train_dataset), len(val_dataset), len(test_dataset)))

        return train_views, val_views, test_views

    def _split_patients_sleeptf(self, dirs_train_whole):
        folds_file = self.config.dataset.data_split.get(
            "folds_file",
            str(_ASSETS_DIR / "data_split_eval.mat"))
        f = loadmat(folds_file)
        f["train_sub"] = f["train_sub"].squeeze() - 1
        f["eval_sub"]  = f["eval_sub"].squeeze()  - 1
        f["test_sub"]  = f["test_sub"].squeeze()  - 1

        train_views = copy.deepcopy(dirs_train_whole)
        val_views   = copy.deepcopy(dirs_train_whole)
        test_views  = copy.deepcopy(dirs_train_whole)

        for view in dirs_train_whole:
            num_difference, prev = 0, -1
            train_dataset, val_dataset, test_dataset = [], [], []
            for file in dirs_train_whole[view]["dataset"]:
                patient_num = int(file["filename"].split("/")[-1][1:5])
                num_difference += patient_num - prev - 1
                prev = patient_num
                idx = patient_num - num_difference
                if idx in f["train_sub"]:
                    train_dataset.append(file)
                elif idx in f["eval_sub"]:
                    val_dataset.append(file)
                elif idx in f["test_sub"]:
                    test_dataset.append(file)
                else:
                    raise Warning("Patient {} has no split assignment.".format(file))
            train_views[view]["dataset"] = train_dataset
            val_views[view]["dataset"]   = val_dataset
            test_views[view]["dataset"]  = test_dataset

        return train_views, val_views, test_views


class Normalization_finder:
    def __init__(self, dataloader, config):
        self.config = config
        self.dataloader = dataloader

    def load_metrics(self):
        norm_dir = self.config.dataset.get(
            "norm_dir",
            str(_ASSETS_DIR / "metrics_eeg_eog_emg_stft.pkl"))
        logging.info("Loading normalisation metrics from {}".format(norm_dir))
        with open(norm_dir, "rb") as f:
            self.metrics = pickle.load(f)

    def get_norm_metrics(self):
        self.load_metrics()
        self.dataloader.train_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.valid_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.total_loader.dataset.set_mean_std(self.metrics["mean"], self.metrics["std"])
        self.dataloader.test_loader.dataset.set_mean_std(self.metrics["mean"],  self.metrics["std"])
        return self.metrics

    def load_metrics_ongoing(self, metrics):
        mean, std = metrics["mean"], metrics["std"]
        self.metrics = metrics
        self.dataloader.train_loader.dataset.set_mean_std(mean, std)
        self.dataloader.valid_loader.dataset.set_mean_std(mean, std)
        self.dataloader.total_loader.dataset.set_mean_std(mean, std)
        self.dataloader.test_loader.dataset.set_mean_std(mean, std)
