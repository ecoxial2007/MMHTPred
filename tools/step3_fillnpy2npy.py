import numpy as np
import glob
import re
import os
import h5py
from tqdm import tqdm

def load_and_sort_npy_files(folder_path):
    # 查找所有的.npy文件
    files = glob.glob(os.path.join(folder_path, 'CT_t*.npy'))
    # files = glob.glob(os.path.join('./data/LARC_center2/feature_temp/JDB016283/', 'CT_t*.npy'))
    # 用文件名中的数字进行排序
    files.sort(key=lambda x: int(re.search("t(\d+)", x).group(1)))

    # 获取文件数量以初始化数组
    num_files = len(files)
    # 假设每个文件的维度是 1 x 256 x 64 x 64
    stacked_array = np.zeros((num_files, 256, 64, 64), dtype=np.float32)

    # 逐个读取文件并填充到数组中
    for i, file in enumerate(files):
        with h5py.File(file, 'r') as f:
            data = f['ct'][:]  # 假设数据集的名称是 'ct'
            stacked_array[i] = data


    return stacked_array


k=0
folder_paths = glob.glob(os.path.join('./data/LARC/feature_temp', 'J*'))

#folder_paths = glob.glob(os.path.join('./feature_temp', 'V*'))
#print(folder_paths)
for folder_path in tqdm(folder_paths):
    # 加载并排序文件
    save_folder = os.path.join(folder_path.replace('/feature_temp/', f'/temp_crop/'), "CT_feature.npy")
    sorted_npy = load_and_sort_npy_files(folder_path)

    # 确保最终数组的形状是 (t, 256, 64, 64)
    print(sorted_npy.shape, save_folder)
    np.save(save_folder, sorted_npy)

