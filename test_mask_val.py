import os
import glob
import random
import argparse
import shutil
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import join
from tools.metrics import ClassificationMetrics
import logging
from tabulate import tabulate
import json
from loss import *
from model_new_mask import HTModel
from dataset_new_mask import H5Dataset
from utils import generate_mask_suffix, move_tensors_to_gpu, count_num_mask
import re

parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="HT-2Class")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-batch_size", type=int, default=8)
# Optimizer parameters
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument('--data_root', type=str, default='data/LARC/temp_crop', help='Root directory of the data')
parser.add_argument('--mask_channels', type=int, default=1, help='Mask channels')
parser.add_argument('--dose_channels', type=int, default=1, help='Dose channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--meta_dim', type=int, default=2, help='Dimension of meta information (e.g., gender, age)')
parser.add_argument('--blood_dim', type=int, default=4, help='Dimension of blood information')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=16, help='Number of heads in attention mechanism')
parser.add_argument('--seq_len', type=int, default=48, help='Sequence length')


parser.add_argument("--fold", type=str, default="0")
parser.add_argument('--use_blood_embedding', type=str, choices=['meta', 'blood', 'meta+blood', 'None'], help='Embedding to use')
parser.add_argument('--use_ptv_mask', action="store_true")
parser.add_argument('--use_bm_mask', action="store_true")
parser.add_argument('--use_fh_mask', action="store_true")
parser.add_argument('--use_ub_mask', action="store_true")
parser.add_argument("--method", type=str, default="doses", choices=['doses', 'images', 'doses+images', 'None'], help="method")

parser.add_argument('--loss_function', type=str, choices=['focal_loss', 'cosface', 'arcface', 'cross_entropy'],
                    default='focal_loss', help='Loss function to use')


args = parser.parse_args()


run_id = datetime.now().strftime("%Y%m%d-%H%M")
blood_suffix = args.use_blood_embedding
mask_suffix = generate_mask_suffix(args)
args.num_mask = count_num_mask(args)

model_save_path = join(args.work_dir,
                       f"{args.task_name}-Val",
                       f"{args.method}-{blood_suffix}-{mask_suffix}-{args.loss_function}",
                       f"fold{args.fold}-{run_id}")

args.model_save_path = model_save_path
device = torch.device(args.device)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
logging.basicConfig(filename=os.path.join(model_save_path, 'testing.log'), level=logging.INFO)
for arg, value in vars(args).items():
    logging.info(f"{arg}: {value}")

def blood_de_norm(name, x_norm):
    # 定义每个血液指标的最大值和最小值
    PLT_max = 512
    PLT_min = 61
    Neut_max = 16.61
    Neut_min = 0.74
    Hb_max = 186
    Hb_min = 59
    WBC_max = 19.62
    WBC_min = 2.06

    # 根据指标名称进行逆归一化
    if name == 'PLT':
        x = x_norm * (PLT_max - PLT_min) + PLT_min
    elif name == 'Neut':
        x = x_norm * (Neut_max - Neut_min) + Neut_min
    elif name == 'Hb':
        x = x_norm * (Hb_max - Hb_min) + Hb_min
    elif name == 'WBC':
        x = x_norm * (WBC_max - WBC_min) + WBC_min
    else:
        x = None

    return x


def get_best_model_roc(resume_path):
    # 用来存储模型和roc图的文件名及其auc值
    models_rocs = []
    # 遍历目录中的文件
    for filename in os.listdir(resume_path):
        # 检查文件名是否符合checkpoint模型的格式
        match_model = re.match(r'model_ep(\d+)_auc([\d.]+)_max_acc([\d.]+).*\.pth', filename)

        if match_model:
            auc_value = match_model.group(2)
            roc_filename = f'{filename.split("_")[1]}_auc{auc_value}.png'
            if roc_filename in os.listdir(resume_path):
                models_rocs.append((filename, roc_filename, float(auc_value)))

    # 按照AUC值排序，从高到低
    models_rocs.sort(key=lambda x: x[2], reverse=True)
    return models_rocs


# %% set up model
def main():
    model = HTModel(args).to(device)
    print("Number of total parameters: ", sum(p.numel() for p in model.parameters()))

    args.meta_file = f'./data/Annotations/fold{args.fold}'

    test_dataset = H5Dataset('val', args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ht_metrics = ClassificationMetrics()

    resume_path = glob.glob(os.path.join(args.work_dir,
                                    f"{args.task_name}-Trainval",
                                    f"{args.method}-{blood_suffix}-{mask_suffix}-{args.loss_function}",
                                    f"fold{args.fold}-*"))[0]

    ckpt, valroc, valauc = get_best_model_roc(resume_path)[0]

    src_valroc = os.path.join(resume_path, valroc)
    dst_valroc = os.path.join(model_save_path, 'val_'+valroc)
    shutil.copy(src_valroc, dst_valroc)

    checkpoint = torch.load(os.path.join(resume_path, ckpt), map_location=device)
    model.load_state_dict(checkpoint["model"])

    logging.info('Start Evaluation ...')

    val_correct = 0
    val_samples = 0
    all_predicted = []
    all_pred = []
    all_ht_label = []
    result_dict = {}

    model.eval()
    with torch.no_grad():
        for step, batch_dict in enumerate(test_dataloader):
            batch_dict = move_tensors_to_gpu(batch_dict, device)

            ht_label = batch_dict['ht_label']
            pred = model(batch_dict)
            logits = F.softmax(pred)
            _, predicted = torch.max(pred, 1)

            print(predicted, ht_label)
            correct = (predicted == ht_label).sum().item()
            val_correct += correct
            val_samples += ht_label.size(0)

            all_predicted.append(predicted)
            all_pred.append(logits)
            all_ht_label.append(ht_label)

            for nid, name in enumerate(batch_dict['folder']):
                blood = batch_dict['blood']
                meta = batch_dict['meta']
                result_dict[name] = {
                    'probability': logits[nid].cpu().tolist(),
                    'ht_label': ht_label[nid].cpu().item(),
                    'ht_pred_0.5': predicted[nid].cpu().item(),
                    'age': meta[nid][0].cpu().item() * 100,
                    'gender': meta[nid][1].cpu().item(),
                    "PLT": blood_de_norm("PLT", blood[nid][0].cpu().item()),
                    "Neut": blood_de_norm("Neut", blood[nid][1].cpu().item()),
                    "Hb": blood_de_norm("Hb", blood[nid][2].cpu().item()),
                    "WBC": blood_de_norm("WBC", blood[nid][3].cpu().item()),
                }


        all_pred = torch.cat(all_pred, dim=0)[:, 1]
        all_predicted = torch.cat(all_predicted, dim=0)
        all_ht_label = torch.cat(all_ht_label, dim=0)

        metrics = ht_metrics.print_metrics(all_predicted, all_ht_label)
        logging.info(str(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid")))

        with open(os.path.join(model_save_path, f'val_result.json'), 'w') as jsonf:
            json.dump(result_dict, jsonf, indent=4)


        fpr, tpr, thresholds = roc_curve(all_ht_label.cpu().tolist(), all_pred.cpu().tolist())
        roc_auc = auc(fpr, tpr)
        # Find optimal threshold
        P = sum(all_ht_label.cpu().tolist())
        N = len(all_ht_label.cpu().tolist()) - P
        # 计算对于每个阈值的准确率
        accuracies = (tpr * P + (1 - fpr) * N) / (P + N)
        # 找到最高准确率及其对应的阈值
        max_accuracy_idx = np.argmax(accuracies)
        max_accuracy = accuracies[max_accuracy_idx]
        optimal_threshold = thresholds[max_accuracy_idx]

        logging.info(
            f'AUC:{roc_auc}, Max Accuracy: {max_accuracy:.4f}, Optimal Threshold: {optimal_threshold:.4f}')
        logging.info(str(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid")))

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(join(model_save_path, f'val_auc{roc_auc:.6f}.png'))


if __name__ == "__main__":
    main()




















































































