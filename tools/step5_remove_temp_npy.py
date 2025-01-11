import os
import shutil
import glob
from tqdm import tqdm

# # 定义要删除的文件夹路径
# folders_to_delete = glob.glob('./data/LARC_center2/temp_dilate/J*/CT') + glob.glob('./data/LARC_center2/temp_dilate/J*/Dose')
# folders_to_delete = glob.glob('./data/LARC/temp_crop/V*/CT') + glob.glob('./data/LARC/temp_crop/V*/Dose')

rm_list = ['CT', 'Dose', 'PTV', 'BoneMarrow', 'FemoralHead', 'UrinaryBladder']
for rm_folder in rm_list:
    folders_to_delete = glob.glob(f'./data/LARC_center2/temp_crop/J*/{rm_folder}')
    # 删除每个文件夹及其内容
    for folder in tqdm(folders_to_delete):
        if os.path.exists(folder):
            shutil.rmtree(folder)

# 定义要删除的文件
# files_to_delete = glob.glob('./data/LARC/temp/V*/CT_feature.npy')# + glob.glob('./data/LARC/temp/V*/Dose.npy')
#
# # 删除每个文件
# for file in tqdm(files_to_delete):
#     if os.path.exists(file):
#         os.remove(file)
