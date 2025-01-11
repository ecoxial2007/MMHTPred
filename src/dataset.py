
import torch
import json
from os.path import join
import numpy as np
from torch.utils.data import Dataset, DataLoader



class H5Dataset(Dataset):
    def __init__(self, split, args):
        self.data_root = args.data_root
        self.meta_file = join(args.meta_file, split + '.json')
        self.num_sample_slide = args.seq_len
        self.args = args

        self.HT_label = {
            0: [0, 1],
            1: [2, 3, 4]
        }

        self.global_max = 6696
        self.global_min = 0
        self.age_max = 100
        self.sample_stride = 1
        self.PLT_max = 512
        self.PLT_min = 61
        self.Neut_max = 16.61
        self.Neut_min = 0.74
        self.Hb_max = 186
        self.Hb_min = 59
        self.WBC_max = 19.62
        self.WBC_min = 2.06

        with open(self.meta_file, 'r') as file:
            self.meta_data = json.load(file)

        self.folder_names = list(self.meta_data.keys())

        print(f"Number of {split} samples: {len(self.folder_names)}")


    def __len__(self):
        return len(self.folder_names)

    def _normalize_min_max(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def _normalize_metric(self, value, metric_min, metric_max):
        return self._normalize_min_max(value, metric_min, metric_max)

    def _process_dose(self, dose_file):
        dose_data = np.load(dose_file)
        if dose_data.shape[-1] != dose_data.shape[-2]:
            dose_data = np.transpose(dose_data, (2, 0, 1))
        normalized_dose = (dose_data - self.global_min) / (self.global_max - self.global_min)
        normalized_dose = np.clip(normalized_dose, 0, 1)
        return normalized_dose



    def _process_meta(self, meta):
        age_normalized = float(meta['age']) / self.age_max
        gender_encoded = int(meta['gender'])
        ht = int(meta['HT'])
        for key, value in self.HT_label.items():
            if ht in value:
                ht_encoded = key
                break
        return age_normalized, gender_encoded, ht_encoded
    def _process_blood(self, meta):
        try:
            PLT = float(meta["PLT#"])
            Neut = float(meta["Neut#"])
            WBC = float(meta["WBC#"])
        except:
            PLT = float(meta["PLT"])
            Neut = float(meta["Neut"])
            WBC = float(meta["WBC"])

        Hb = float(meta["Hb"])

        normalized_PLT = self._normalize_metric(PLT, self.PLT_min, self.PLT_max)
        normalized_Neut = self._normalize_metric(Neut, self.Neut_min, self.Neut_max)
        normalized_Hb = self._normalize_metric(Hb, self.Hb_min, self.Hb_max)
        normalized_WBC = self._normalize_metric(WBC, self.WBC_min, self.WBC_max)

        return normalized_PLT, normalized_Neut, normalized_Hb, normalized_WBC

    def get_sample_indices(self, seq_length):
        mid_index = seq_length // 2
        half_samples = self.num_sample_slide // 2
        start_index = max(0, mid_index - half_samples * self.sample_stride)
        end_index = min(seq_length, mid_index + half_samples * self.sample_stride)

        if start_index == 0:
            sample_indices = np.arange(start_index, start_index + self.num_sample_slide * self.sample_stride,
                                       self.sample_stride)
        else:
            sample_indices = np.arange(end_index - self.num_sample_slide * self.sample_stride, end_index,
                                       self.sample_stride)

        return sample_indices

    def get_mask(self, result_dict):
        if self.args.use_ptv_mask:
            result_dict['ptv'] = self._load_and_process_mask(result_dict['folder'], 'PTV.npy')
        if self.args.use_bm_mask:
            result_dict['bm'] = self._load_and_process_mask(result_dict['folder'], 'BoneMarrow.npy')
        if self.args.use_fh_mask:
            result_dict['fh'] = self._load_and_process_mask(result_dict['folder'], 'FemoralHead.npy')
        if self.args.use_ub_mask:
            result_dict['ub'] = self._load_and_process_mask(result_dict['folder'], 'UrinaryBladder.npy')
        return result_dict

    def _load_and_process_mask(self, folder_name, file_name):
        file_path = join(self.data_root, folder_name, file_name)
        mask = np.load(file_path) / 255.0

        T, H, W = mask.shape
        mask_zero = np.zeros((self.num_sample_slide, H, W))

        if T < self.num_sample_slide:
            start_idx = (self.num_sample_slide - T) // 2
            mask_zero[start_idx:start_idx + T] = mask

        else:
            mid_idx = T // 2
            half_samples = self.num_sample_slide // 2
            start_idx = max(0, mid_idx - half_samples)
            end_idx = min(T, mid_idx + half_samples)
            mask_zero[:end_idx - start_idx] = mask[start_idx:end_idx]

        return torch.tensor(mask_zero, dtype=torch.float)

    def __getitem__(self, index):
        folder_name = self.folder_names[index]
        meta = self.meta_data[folder_name]
        age, gender, ht_label = self._process_meta(meta)
        result_dict = {
            'folder': folder_name,
            'ht_label': torch.tensor(ht_label, dtype=torch.long)
        }

        normalized_PLT, normalized_Neut, normalized_Hb, normalized_WBC = self._process_blood(meta)

        if self.args.method != 'None':
            img_file = join(self.data_root, folder_name, "CT_feature.npy")
            dose_file = join(self.data_root, folder_name, "Dose.npy")
            dose = self._process_dose(dose_file)
            img = np.load(img_file)
            T = img.shape[0]
            img_zero = np.zeros((self.num_sample_slide, 256, img.shape[2], img.shape[3]))
            dose_zero = np.zeros((self.num_sample_slide, dose.shape[1], dose.shape[2]))

            if T < self.num_sample_slide:
                start_idx = (self.num_sample_slide - T) // 2
                img_zero[start_idx:start_idx + T] = img
                dose_zero[start_idx:start_idx + T] = dose
            else:
                mid_idx = T // 2
                half_samples = self.num_sample_slide // 2
                start_idx = max(0, mid_idx - half_samples)
                end_idx = min(T, mid_idx + half_samples)
                img_zero[:end_idx - start_idx] = img[start_idx:end_idx]
                dose_zero[:end_idx - start_idx] = dose[start_idx:end_idx]

            result_dict['images'] = torch.tensor(img_zero, dtype=torch.float)
            result_dict['doses'] = torch.tensor(dose_zero, dtype=torch.float)
            result_dict = self.get_mask(result_dict)


        blood = torch.tensor([normalized_PLT, normalized_Neut, normalized_Hb, normalized_WBC], dtype=torch.float)
        meta = torch.tensor([age, gender], dtype=torch.float)

        result_dict['blood'] = blood
        result_dict['meta'] = meta

        return result_dict
