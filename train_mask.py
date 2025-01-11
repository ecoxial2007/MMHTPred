import os
from os.path import join
import argparse
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tools.metrics import ClassificationMetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

from src.loss import *
from src.model import HTModel
from src.dataset import H5Dataset
from src.utils import generate_mask_suffix, move_tensors_to_gpu, count_num_mask

parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="HT-2Class")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=15)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=1e-4, metavar="LR", help="1e-4 learning rate (absolute lr)")
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
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
blood_suffix =  args.use_blood_embedding
mask_suffix = generate_mask_suffix(args)
args.num_mask = count_num_mask(args)

model_save_path = join(args.work_dir,
                       f"{args.task_name}-Trainval",
                       f"{args.method}-{blood_suffix}-{mask_suffix}-{args.loss_function}",
                       f"fold{args.fold}-{run_id}")

args.model_save_path = model_save_path

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

device = torch.device(args.device)
logging.basicConfig(filename=os.path.join(model_save_path, 'testing.log'), level=logging.INFO)
for arg, value in vars(args).items():
    logging.info(f"{arg}: {value}")

# %% set up model
def main():
    model = HTModel(args).to(device)
    print("Number of total parameters: ", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss_function == 'focal_loss':
        criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    elif args.loss_function == 'arcface':
        criterion = ArcFaceLoss(s=30.0, m=0.50)
    elif args.loss_function == 'cosface':
        criterion = CosFaceLoss(s=30.0, m=0.35)
    else:
        criterion = nn.CrossEntropyLoss()


    num_epochs = args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)  # 创建一个余弦退火学习率调度器
    iter_num = 0
    losses = []
    args.meta_file = f'./data/Annotations/fold{args.fold}'

    train_dataset = H5Dataset('train', args)  # split =train val test
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = H5Dataset('val', args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    ht_metrics = ClassificationMetrics()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()


    start_epoch = 0
    best_auc = 0
    for epoch in range(start_epoch, num_epochs):
        if epoch == 6: quit()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0
        epoch_start_time = datetime.now()
        model.train()
        for step, batch_dict in enumerate(train_dataloader):


            step_start_time = datetime.now()
            optimizer.zero_grad()
            batch_dict = move_tensors_to_gpu(batch_dict, device)
            ht_label = batch_dict['ht_label']
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(batch_dict)
                    loss = criterion(pred, ht_label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                pred = model(batch_dict)
                loss = criterion(pred, ht_label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            _, predicted = torch.max(pred, 1)  # ce
            # print(predicted, ht_label)
            correct = (predicted == ht_label).sum().item()
            acc = correct / ht_label.size(0)

            epoch_loss += loss.item()
            total_correct += correct
            total_samples += ht_label.size(0)
            iter_num += 1
            step_end_time = datetime.now()
            step_duration = (step_end_time - step_start_time).total_seconds()
            logging.info(
                f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, LR: {args.lr}, Step: {step}/{len(train_dataloader)}, Loss: {loss.item()}, Current Accuracy: {acc}, Time: {step_duration}')

        epoch_loss /= max(step, 1)
        train_accuracy = total_correct / total_samples
        losses.append(epoch_loss)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        logging.info(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}, Total Accuracy: {train_accuracy}, Time: {epoch_duration}'
        )

        if epoch<2:continue
        logging.info('Start Evaluation ...')
        val_loss = 0
        val_correct = 0
        val_samples = 0
        all_predicted = []
        all_pred = []
        all_ht_label = []
        model.eval()
        with torch.no_grad():
            for step, batch_dict in enumerate(val_dataloader):
                batch_dict = move_tensors_to_gpu(batch_dict, device)
                ht_label = batch_dict['ht_label']
                pred = model(batch_dict)
                loss = criterion(pred, ht_label)
                logits = F.softmax(pred)
                _, predicted = torch.max(pred, 1)

                print(predicted, ht_label)
                correct = (predicted == ht_label).sum().item()
                val_correct += correct
                val_samples += ht_label.size(0)
                val_loss += loss.item()

                all_predicted.append(predicted)
                all_pred.append(logits)
                all_ht_label.append(ht_label)
                logging.info(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Step: {step}/{len(val_dataloader)}, Loss: {loss.item()}, Current Accuracy: {val_correct / val_samples}')

            val_accuracy = val_correct / val_samples
            all_pred = torch.cat(all_pred, dim=0)[:, 1]
            all_predicted = torch.cat(all_predicted, dim=0)
            all_ht_label = torch.cat(all_ht_label, dim=0)
            metrics = ht_metrics.print_metrics(all_predicted, all_ht_label)
            val_sensitive = metrics[1][1]
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

            logging.info(f'Epoch {epoch}, Validation loss: {val_loss / len(val_dataloader)}, Validation accuracy: {val_accuracy}, AUC:{roc_auc}, Max Accuracy: {max_accuracy:.4f}, Optimal Threshold: {optimal_threshold:.4f}')
            logging.info(str(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid")))

            ## save the latest model
            model_save_path_epoch = os.path.join(model_save_path, f"model_ep{epoch}_auc{roc_auc:.6f}_max_acc{max_accuracy:.4f}_Threshold{optimal_threshold:.4f}.pth")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_sensitive": val_sensitive,
                "val_acc": val_accuracy,
                "epoch": epoch,
                "roc_auc": roc_auc
            }

            # 保存最佳 AUC 模型
            # if roc_auc >= best_auc:
            #     best_auc = roc_auc
            torch.save(checkpoint, model_save_path_epoch)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(join(model_save_path, f'ep{epoch}_auc{roc_auc:.6f}.png'))

        scheduler.step()


if __name__ == "__main__":
    main()




















































































