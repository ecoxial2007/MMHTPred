import json
import nibabel as nib
import numpy as np
import os
import glob
from tqdm import tqdm
import cv2

HEIGHT, WIDTH = 256, 256


def find_min_max_median(img_data, dose_data, all_masks, region_names, t):
    stats = []
    for region_name, mask_data in zip(region_names, all_masks):
        # 获得当前切片的掩码
        mask_slice = mask_data[:, :, t]

        # 应用掩码到CT和剂量数据
        img_slice = img_data[:, :, t] * (mask_slice > 0)
        dose_slice = dose_data[:, :, t] * (mask_slice > 0)

        # 初始化统计数据字典
        stat = {
            'region_name': region_name,
            'max_img': 0,
            'min_img': 0,
            'median_img': 0,
            'max_dose': 0,
            'min_dose': 0,
            'median_dose': 0
        }

        # 检查并计算统计数据，避免空数组错误
        if img_slice.size > 0 and np.any(img_slice):  # 检查数组非空且有非零元素
            stat['max_img'] = np.max(img_slice)
            stat['min_img'] = np.min(img_slice)
            stat['median_img'] = np.median(img_slice)

        if dose_slice.size > 0 and np.any(dose_slice):
            try:
                stat['max_dose'] = np.max(dose_slice)
                stat['min_dose'] = np.min(dose_slice)
                stat['median_dose'] = np.median(dose_slice)
            except ValueError as e:
                print(f"Error processing {region_name} at slice {t}: {e}")

        stats.append(stat)

    return stats


def crop_file_save(filepath, temp_folder='./tools/stat/'):
    region_names = ['Body', 'PTV', 'BoneMarrow', 'FemoralHead', 'UrinaryBladder']
    filenames = glob.glob(os.path.join(filepath, '*', "FemoralHead.nii.gz"))


    for f in tqdm(filenames):
        folder_name = f.split('/')[-2]
        sssave_path = os.path.join(temp_folder, f'{folder_name}_statistics.json')
        if os.path.exists(sssave_path):continue
        all_stats = []
        img_path = f.replace("FemoralHead.nii.gz", 'CT.nii.gz')
        dose_path = img_path.replace('CT.nii.gz', 'Dose.nii.gz')

        img = nib.load(img_path)
        img_data = img.get_fdata()

        dose = nib.load(dose_path)
        dose_data = dose.get_fdata()

        all_masks = []
        for region_name in region_names:
            mask_path = img_path.replace('CT.nii.gz', f'{region_name}.nii.gz')
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata()
            all_masks.append(mask_data)

        _, _, T = img_data.shape
        for t in range(T):
            stats = find_min_max_median(img_data, dose_data, all_masks, region_names, t)
            if stats != []:
                all_stats.append(stats)

        # 将统计数据保存到 JSON 文件
        with open(os.path.join(temp_folder, f'{folder_name}_statistics.json'), 'w') as json_file:
            json.dump(all_stats, json_file, indent=4)


# 使用示例
filepath = './data/LARC/source/'  # 修改为你的文件路径
crop_file_save(filepath)
