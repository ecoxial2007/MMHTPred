import nibabel as nib
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import cv2

HEIGHT, WIDTH = 256, 256

def crop_and_save(img_data, all_mask, region_names, dose_data, t, temp_folder, min_x1, min_y1, max_x2, max_y2):

    mask_slice = all_mask[0][:, :, t]
    img_slice = img_data[:, :, t]
    dose_slice = dose_data[:, :, t]

    if np.any(mask_slice):
        # 获取边界坐标
        rows = np.any(mask_slice, axis=1)
        cols = np.any(mask_slice, axis=0)
        y, x = np.where(rows)[0][[0, -1]], np.where(cols)[0][[0, -1]]

        min_x1, min_y1 = min(min_x1, x[0]), min(min_y1, y[0])
        max_x2, max_y2 = max(max_x2, x[1]), max(max_y2, y[1])

    # else:
    #     img_slice = np.zeros_like(img_slice)
    #     dose_slice = np.zeros_like(dose_slice)

    for region_name, each_mask in zip(region_names, all_mask):
        cv2.imwrite(os.path.join(temp_folder, f'{region_name}/{t}.png'), 255 * each_mask[:, :, t])

    cv2.imwrite(os.path.join(temp_folder, f'CT/{t}.png'), img_slice)
    np.save(os.path.join(temp_folder, f'Dose/{t}.npy'), dose_slice)

    return min_x1, min_y1, max_x2, max_y2

def pad_and_resize(image, target_size, padding_value=0):
    """将图像填充为正方形并调整大小到目标尺寸"""
    height, width = image.shape[:2]
    max_side = max(height, width)

    # 计算填充大小
    delta_w = max_side - width
    delta_h = max_side - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 为二维图像进行填充
    padded_img = np.pad(image, ((top, bottom), (left, right)), 'constant', constant_values=padding_value)

    # 调整图像大小到目标尺寸
    resized_img = cv2.resize(padded_img, (target_size, target_size))

    return resized_img

def crop_file_save(filepath, region_name='PTV'):
    filenames = glob.glob(os.path.join(filepath, '*', 'FemoralHead.nii.gz'))

    region_names = ['PTV', 'UrinaryBladder', 'BoneMarrow', 'FemoralHead']
    for f in tqdm(filenames):
        img_path = f.replace('FemoralHead.nii.gz', 'CT.nii.gz')
        dose_path = img_path.replace('CT.nii.gz', 'Dose.nii.gz')

        temp_folder = img_path.replace('CT.nii.gz', '').rstrip('/')
        temp_folder = temp_folder.replace('source', f'temp_crop')

        os.makedirs(os.path.join(temp_folder, 'CT'), exist_ok=True)
        os.makedirs(os.path.join(temp_folder, 'Dose'), exist_ok=True)



        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255

        dose = nib.load(dose_path)
        dose_data = dose.get_fdata()

        all_mask = []
        for region_name in region_names:
            os.makedirs(os.path.join(temp_folder, region_name), exist_ok=True)
            mask_path = img_path.replace('CT.nii.gz', f'{region_name}.nii.gz')

            mask = nib.load(mask_path)
            mask_data = mask.get_fdata()
            all_mask.append(mask_data)


        _, _, T = mask_data.shape
        min_x1, min_y1, max_x2, max_y2 = 999, 999, 0, 0

        for t in range(T):
            min_x1, min_y1, max_x2, max_y2 = crop_and_save(img_data,
                                                           all_mask, region_names,
                                                           dose_data,
                                                           t, temp_folder, min_x1, min_y1, max_x2, max_y2)

        ct_all = []
        dose_all = []

        for t in range(T):
            ct_path = os.path.join(temp_folder, f'CT/{t}.png')
            dose_path = os.path.join(temp_folder, f'Dose/{t}.npy')

            ct_img = Image.open(ct_path)
            dose_img = np.load(dose_path)

            cropped_ct_img = ct_img.crop((min_x1, min_y1, max_x2, max_y2))
            cropped_dose_img = dose_img[min_y1: max_y2, min_x1: max_x2]

            # Pad images to make them square
            max_side = max(cropped_ct_img.size)
            delta_w = max_side - cropped_ct_img.size[0]
            delta_h = max_side - cropped_ct_img.size[1]
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            ct_img_padded = ImageOps.expand(cropped_ct_img, padding)
            # dose_img_padded = ImageOps.expand(cropped_dose_img, padding)
            dose_img_resized = pad_and_resize(cropped_dose_img, HEIGHT, WIDTH)

            # Resize images
            ct_img_resized = ct_img_padded.resize((WIDTH*2, HEIGHT*2))
            ct_all.append(np.array(ct_img_resized))
            dose_all.append(np.array(dose_img_resized))


        print(temp_folder)
        np.save(os.path.join(temp_folder, f'CT.npy'), np.array(ct_all))
        np.save(os.path.join(temp_folder, f'Dose.npy'), np.array(dose_all))


        for region_name in region_names[1:]:
            mask_all = []
            for t in range(T):
                mask_path = os.path.join(temp_folder, f'{region_name}/{t}.png')
                mask_img = Image.open(mask_path)
                cropped_mask_img = mask_img.crop((min_x1, min_y1, max_x2, max_y2))

                # Pad images to make them square
                max_side = max(cropped_mask_img.size)
                delta_w = max_side - cropped_mask_img.size[0]
                delta_h = max_side - cropped_mask_img.size[1]
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                mask_img_padded = ImageOps.expand(cropped_mask_img, padding)

                # Resize images
                mask_img_resized = mask_img_padded.resize((WIDTH * 2, HEIGHT * 2))
                mask_all.append(np.array(mask_img_resized))
            print(temp_folder, region_name)
            np.save(os.path.join(temp_folder, f'{region_name}.npy'), np.array(mask_all))




# 使用示例
filepath = './data/LARC/source/'  # 修改为你的文件路径
crop_file_save(filepath)
