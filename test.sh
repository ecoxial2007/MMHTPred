#!/bin/bash

# 固定掩码参数，根据需要在这里手动开启或关闭
PTV_MASK="--use_ptv_mask"  # 使用 "--use_ptv_mask" 或 ""（空表示不使用）--use_ptv_mask
BM_MASK=""    # 使用 "--use_bm_mask" 或 ""
FH_MASK=""    # 使用 "--use_fh_mask" 或 ""
UB_MASK=""    # 使用 "--use_ub_mask" 或 ""


# 设置 fold 和 method 的可能值
folds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

methods=("doses" "images" "doses+images") #None
embeddings=("None" "None""meta+blood" "meta")

# 遍历 embeddings
for embedding in "${embeddings[@]}"; do
    for fold in "${folds[@]}"; do
      for method in "${methods[@]}"; do
        # 遍历 folds
            echo "Running with fold=$fold, embedding=$embedding, method=$method, ptv=$PTV_MASK, bm=$BM_MASK, fh=$FH_MASK, ub=$UB_MASK"
            python test_mask.py -task_name "HT-2Class-Test" --fold "$fold" --use_blood_embedding "$embedding" $PTV_MASK $BM_MASK $FH_MASK $UB_MASK --method "$method"
            python test_mask_val.py -task_name "HT-2Class-Val" --fold "$fold" --use_blood_embedding "$embedding" $PTV_MASK $BM_MASK $FH_MASK $UB_MASK --method "$method"
            python test_mask_ext.py -task_name "HT-2Class-ExtTest" --fold "$fold" --use_blood_embedding "$embedding" $PTV_MASK $BM_MASK $FH_MASK $UB_MASK --method "$method"
        done
    done
done
