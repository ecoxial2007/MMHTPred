import os
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm


k = 0
folders = glob.glob(f'./data/LARC/temp_crop/*')

min_t = 999
region_names = ["UrinaryBladder", 'BoneMarrow', 'FemoralHead']
for folder in tqdm(folders):
    ptv_path = os.path.join(folder.replace(f'/temp_crop/', '/source/'), 'PTV.nii.gz')
    ct_path = os.path.join(folder, 'CT.npy')
    ctfea_path = os.path.join(folder, 'CT_feature.npy')
    dose_path = os.path.join(folder, 'Dose.npy')
    region_paths = [os.path.join(folder, region_name+'.npy') for region_name in region_names]

    mask = nib.load(ptv_path).get_fdata()
    ct = np.load(ct_path)
    ctfea = np.load(ctfea_path)
    dose = np.load(dose_path)
    regions = [np.load(region_path) for region_path in region_paths]

    non_zero_slices = np.any(mask, axis=(0, 1))
    # 找到第一个和最后一个非全零切片的索引
    first_non_zero_slice = np.argmax(non_zero_slices)
    last_non_zero_slice = len(non_zero_slices) - np.argmax(non_zero_slices[::-1]) - 1

    ct_crop = ct[first_non_zero_slice:last_non_zero_slice + 1, :, :]
    ctfea_crop = ctfea[first_non_zero_slice:last_non_zero_slice + 1, :, :, :]
    dose_crop = dose[first_non_zero_slice:last_non_zero_slice + 1, :, :]
    region_crops = [region[first_non_zero_slice:last_non_zero_slice + 1, :, :] for region in regions]

    # t = ct_crop.shape[0]
    t = last_non_zero_slice - first_non_zero_slice + 1
    min_t = min(t, min_t)

    new_ct_path = os.path.join(folder, 'CT.npy')
    new_ctfea_path = os.path.join(folder, 'CT_feature.npy')
    new_dose_path = os.path.join(folder, 'Dose.npy')
    new_region_paths = [os.path.join(folder, region_name+'.npy') for region_name in region_names]


    np.save(new_ct_path, ct_crop)
    np.save(new_dose_path, dose_crop)
    np.save(new_ctfea_path, ctfea_crop)
    for region_crop, new_region_path in zip(region_crops, new_region_paths):
        np.save(new_region_path, region_crop)
        print(region_crop.shape, new_region_path)
    print(ct_crop.shape, new_ct_path)
    print(ctfea_crop.shape, new_ctfea_path)
    print(dose_crop.shape, new_dose_path)
