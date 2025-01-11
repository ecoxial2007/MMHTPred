## Data Organization
This project organizes data in the following structure:
```

data/
├── Annotation/                # Annotations for training, validation, and testing
│   ├── fold0/
│   │   ├── train.json         # Training set annotation
│   │   ├── val.json           # Validation set annotation
│   │   ├── test.json          # Test set annotation
│   ├── test_ext.json          # External test set annotation
├── LARC/                      # Contains patient-specific data
│   ├── source/                # Original CT/Dose/Mask data (.nii.gz)
│   ├── temp_features/         # Extracted features for each patient's CT slices (.npy)
│   ├── temp_crop/             # Cropped CT/Dose/Mask data and Combined Feature (.npy)
│   │   ├── V32505
│   │   │   ├── CT_feature.npy
│   │   │   ├── Dose.npy
│   │   │   ├── PTV.npy
│   │   │   ├── UrinaryBladder.npy
│   │   │   ├── FemoralHead.npy
│   │   │   ├── BoneMarrow.npy
```