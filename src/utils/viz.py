# path: ~/Develop/rotating-machinery-fault-analysis/src/utils/viz.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd

def plot_confusion_matrix(model, x_test, y_test, save_path: Path):
    """혼동 행렬을 그립니다."""
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        y_hat = model(x_test)
        _, predicted = torch.max(y_hat.data, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    classes = ('Normal', 'Type1', 'Type2', 'Type3')
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 5))
    df_cm = pd.DataFrame(
        cm / len(x_test),
        index=[i for i in classes],
        columns=[i for i in classes]
    )
    
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", size=24, fontweight='bold')
    plt.xlabel("Predicted Label", size=16)
    plt.ylabel("Actual Label", size=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.yticks(rotation=0)
    
    plt.savefig(save_path)
    plt.close()

def plot_loss_graph(loss_values, loss_values_v, title, save_path: Path):
    """학습 및 검증 손실 그래프를 그립니다."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.plot(loss_values, label='Train')
    plt.plot(loss_values_v, label='Validation')
    plt.title(title, size=16)
    plt.xlabel("Epoch", size=14)
    plt.ylabel("Loss", size=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()