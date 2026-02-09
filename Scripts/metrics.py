import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def super_cm(
    cm=None, class_name=None,
    scores=None, labels=None,
    x_lim=(0,1), y_lim=(0,1),
    figsize=(6,6), plor_rate=1,
    color = "#AABCDB", title=None,
    family=None, pathway=None):

    if family is not None:
        plt.rcParams['font.family'] = family

    # prepare parameters
    cm = np.array(cm,dtype=np.float64)
    x_min, x_max = x_lim
    x_step = round((x_max - x_min) / 5,2)
    y_min, y_max = y_lim
    y_step = round((y_max - y_min) / 5,2)
    my_cmap = LinearSegmentedColormap.from_list('mycmap', ['#FFFFFF', color], N=256)

    # new fig
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    for name in ['top', 'right', 'left', 'bottom']:
        ax.spines[name].set_visible(False)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 2])

    # subplot 1
    ax1 = fig.add_subplot(gs[0,0])
    for i in np.arange(y_min,y_min+y_step*5,y_step):
        ax1.axhline(y=i, linestyle='dashed', color="black", linewidth=1, alpha=0.1)
    height = cm.max(axis=0) / (cm.sum(axis=0)+1e-10)
    height = [round(i, 3) for i in height]
    bar1 = ax1.bar(x=range(len(cm)), height = height, color=color, alpha=0.5, width=0.7)
    ax1.bar_label(bar1, height, color="Black", alpha=0.4)
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(np.arange(y_min,y_min+y_step*5.1,y_step))
    ax1.set_ylabel("Precision")
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # subplot 2
    ax2 = fig.add_subplot(gs[1,1])
    for i in np.arange(x_min,x_min+x_step*5,x_step):
        ax2.axvline(x=i, linestyle='dashed', color="black", linewidth=1, alpha=0.1)
    width = cm.max(axis=1) / (cm.sum(axis=1)+1e-10)
    width = [round(i, 3) for i in width]
    bar2 = ax2.barh(y=range(len(cm)), width = width, color=color, alpha=0.5, height=0.7)
    ax2.bar_label(bar2, width, color="Black", alpha=0.4, rotation=-90)
    ax2.set_xlim(x_min,x_max)
    ax2.set_xticks(np.arange(x_min,x_min+x_step*5.1,x_step), ["{:.2f}".format(i) for i in np.arange(x_min,x_min+x_step*5.1,x_step)],rotation=-90)
    ax2.set_xlabel("Recall")
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    # subplot 3
    ax3 = fig.add_subplot(gs[1,0])
    cm_plot = np.array([cm[i] / np.sum(cm[i]) for i in range(len(cm))])
    cm_plot = np.where(cm_plot < 0.1, cm_plot*plor_rate, cm_plot)
    plt.imshow(cm_plot, origin='lower', cmap=my_cmap)
    for i in range(cm.shape[0]):
        s=sum(cm[i])
        for j in range(cm.shape[1]):
            ax3.text(j, i, "{:.3f}".format(cm[i][j]/s), ha='center', va='center', color="Black", alpha=0.4)
    bbox3 = ax3.get_position()
    new_bbox3 = [bbox3.x0-0.12, bbox3.y0-0.12, bbox3.width, bbox3.height]
    ax3.set_position(new_bbox3)
    ax3.set_xticks(np.arange(0,len(class_name),1), class_name, rotation=-90)
    ax3.set_yticks(np.arange(0,len(class_name),1), class_name)
    ax3.set_xlabel("Prediction")
    ax3.set_ylabel("Ground Truth")
    
    # suplot 4
    ax4 = fig.add_subplot(gs[0,1])
    fpr, tpr, thresholds = roc_curve(labels, scores[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=color, label=f'ROC (area = {roc_auc:.2f})', lw=1.2)
    ax4.fill_between(fpr, tpr, alpha=0.3, color=color)
    ax4.set_xlim([-0.05, 1.05])
    ax4.set_ylim([-0.05, 1.05])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.annotate(f'AUC = {roc_auc:.3f}', xy=(0.4, 0.1), fontsize=8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if pathway is not None:
        plt.savefig(pathway)
    plt.show()


import os
import pytorch_lightning as pl
from Scripts.model4nb import MMM
import Scripts.config as config
from Scripts.dataset import create_dataloader, transform_us, transform_dp
from Scripts.metrics import super_cm
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torchmetrics as tm
from tqdm import tqdm

def summary(mode = "MT", num_classes = 2, device = "cuda:7", train_set=True):
    checkpoint = os.path.join("checkpoints_Class{}_v3".format(num_classes), mode+"_best.ckpt")

    dataloader = create_dataloader(
            root="Dataset_Class{}".format(num_classes),
            batch_size=config.BATCH_SIZE,
            transform_us=transform_us,
            transform_dp=transform_dp,
            need_filename = True
        )

    model = MMM.load_from_checkpoint(
        checkpoint,
        lr = config.LEARNING_RATE,
        single_decoder = True,
        mode = mode,
        num_classes = num_classes,
        alpha = config.ALPHA,
        gamma = config.GAMMA,
        delta = config.DELTA,
        device = device).eval()
    model = model.to(device)

    scores = []
    labels = []
    filename_list = []
    if train_set:
        for us, dp, tx, mt, label, filename in tqdm(dataloader["train"]):
            label = label.to(device)
            x = {"US": us.to(device),
                "DP": dp.to(device),
                "TX": tx,
                "MT": mt.to(device)}
            outputs = model(x, return_features=False)
            labels.append(label)
            scores.append(outputs)
            for f in filename:
                filename_list.append(f)
    for us, dp, tx, mt, label, filename in tqdm(dataloader["test"]):
        label = label.to(device)
        x = {"US": us.to(device),
            "DP": dp.to(device),
            "TX": tx,
            "MT": mt.to(device)}
        outputs = model(x, return_features=False)
        labels.append(label)
        scores.append(outputs)
        for f in filename:
            filename_list.append(f)
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    auc_tester = tm.AUROC(task="multiclass", num_classes=num_classes)
    auc = auc_tester(scores, labels).item()
    metrics = {"ACC":[], "FPR":[], "TPR":[], "SEN":[],"SPE":[], "AUC":[]}
    _, preds = scores.max(1)
    preds = preds.cpu()

    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    cm = confusion_matrix(labels, preds)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        fpr = round(fp / (fp + tn), 3)
        tpr = round(tp / (tp + fn), 3)
        sen = round(tp / (tp + fn), 3)
        spe = round(tn / (tn + fp), 3)

    elif num_classes > 2:
        n_classes = cm.shape[0]
        tpr_list, fpr_list, sen_list, spe_list = [], [], [], []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)

            tpr_list.append(tp / (tp + fn))
            fpr_list.append(fp / (fp + tn))
            sen_list.append(tp / (tp + fn))
            spe_list.append(tn / (tn + fp))

        fpr = round(np.mean(fpr_list), 3)
        tpr = round(np.mean(tpr_list), 3)
        sen = round(np.mean(sen_list), 3)
        spe = round(np.mean(spe_list), 3)

    metrics["ACC"].append(round(accuracy_score(labels, preds), 3))
    metrics["FPR"].append(fpr)
    metrics["TPR"].append(tpr)
    metrics["SEN"].append(sen)
    metrics["SPE"].append(spe)
    metrics["AUC"].append(round(auc, 3))

    return metrics