
# MMHTPred
Multimodal integration of CT images, dose map, hematologic and demographic features for pre-radiotherapy prediction of hematologic toxicity in rectal cancer


## Project Overview
This project focuses on integrating multimodal data, including CT images, dose maps, hematologic, and demographic features, to build a predictive model for hematologic toxicity (HT) in rectal cancer patients undergoing radiotherapy. 
The pipeline includes CT and dose data preprocessing, feature extraction (MedSAM), and model training/validation, all implemented using the PyTorch framework.

---

## Directory Structure
```
MMHTPred/
├── checkpoints/            # Pre-trained model weights
├── data/                   # Dataset directory
│   ├── Annotation/         # Training/validation/testing/external testing splits
│   ├── LARC/               # CT, Dose, Mask and feature files
├── src/                    # Core scripts for dataset loading, model, and utilities
│   ├── dataset.py          # Data loading module
│   ├── model.py            # Model definition
│   ├── loss.py             # Loss functions
│   ├── utils.py            # Utility functions
├── tools/                  # Scripts for data preprocessing and feature preparation
│   ├── README.md           # Instructions for data pre-processing and feature extraction
│   ├── step0_analysis_dose_median_range.py
│   ├── step1_mask_image_process.py
│   ├── step2_inference_npy.py
│   ├── ...
├── train.sh                # Script to start 10-fold training and validation
├── test.sh                 # Script to start HT prediction(ROC/AUC...) for val/test/external testset
```

---

## Installation
Ensure you have Python 3.8+ installed. Run the following command to install the necessary dependencies:
```bash
pip install torch==2.0.0 scikit-learn matplotlib tabulate einops
```

---

## Data Preparation
1. Refer to `tools/README.md` for detailed instructions on preprocessing and feature preparation. Key steps include:
   - (Optional) `step0_analysis_dose_median_range.py`: Analyze dose distribution statistics. 
   - `step1_mask_image_process.py`: Center crop for CT, dose and mask. 
   - `step2_inference_npy.py`: Generate intermediate feature files using MedSAM (checkpoint in `./checkpoints/medsam-vit-b/medsam_vit_b/medsam_vit_b.pth`).
   - `step3_fillnpy2npy.py`: Merge feature sequences.
   - `step4_check_zero_mask.py`: Select the intermediate sequence based on the mask.
2. Organize the preprocessed data in the `data/` directory as per the required format.

---

## Training and Validation
1. Configure the `train.sh` script to set appropriate parameters for your experiments.
2. Start the 10-fold training and validation process by running:
   ```bash
   bash train.sh
   ```

---

## Test for ROC / AUC / Accuracy
1. Configure the `test.sh` script to set appropriate parameters for your experiments.
2. Start the 10-fold training and validation process by running:
   ```bash
   bash test.sh
   ```
   
Note: All parameter settings in `test.sh`, including the checkpoints path, must be consistent with those in `train.sh`, otherwise the checkpoints cannot be automatically loaded.

After completing the testing phase, the prediction results for each case, along with the ROC curve and evaluation metrics, will be saved in the following directory structure:
```
work_dir/
├── HT-2Class-Test/
│   ├── doses+images-meta+blood-ptv-focal_loss/
│   │   ├── fold0-20240717-1510/
│   │   │   ├── test_auc0.844113.png       # ROC curve visualization
│   │   │   ├── test_result.json           # Prediction results for each case
│   │   │   ├── testing.log               # Testing logs

```

---

## Citation
If you use this project or its components in your research, please cite as follows:
```bibtex
```

---

