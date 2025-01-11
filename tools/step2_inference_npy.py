import os
import glob
import numpy as np
import torch
from skimage import transform
import h5py
from tqdm import tqdm
from segment_anything import sam_model_registry

if __name__ == '__main__':

    MedSAM_CKPT_PATH = "./checkpoints/medsam-vit-b/medsam_vit_b.pth"
    device = "cuda:0"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    root = f'./data/LARC/temp_crop'

    data_paths = glob.glob(os.path.join(root, "*", 'CT.npy'))
    for data_path in tqdm(data_paths):
        if os.path.exists(data_path.replace('CT.npy', 'CT_feature.npy')):
            print(data_path)
            continue

        feat_path = data_path.replace(f'/temp_crop/', '/temp_features/')
        file_path, name = os.path.split(feat_path)

        if not os.path.exists(file_path):
            os.makedirs(file_path)  # 新建文件夹

        ct_images = np.load(data_path)  # Load CT images from .npy file
        for t, img_np in enumerate(ct_images):
            time_feat_path = feat_path.replace('.npy', f'_t{t}.npy')
            if os.path.exists(time_feat_path):continue

            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor)

            numpy_array = image_embedding.cpu().numpy()

            time_feat_path = feat_path.replace('.npy', f'_t{t}.npy')
            print(time_feat_path, image_embedding.shape)
            with h5py.File(time_feat_path, 'w') as f:
                f.create_dataset('ct', data=numpy_array)
