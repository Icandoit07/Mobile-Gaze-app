import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import model
from gazetrack_dataset import gazetrack_dataset

def get_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors

def euc(preds, gt):
    return np.linalg.norm(preds - gt, axis=1)

def main():

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds, gt = [], []


    f = files[idx]
    fs = glob.glob(f + "*.jpg")
    test_dataset = gazetrack_dataset(f, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False)

    for batch in tqdm(test_dataloader):
        leye, reye, kps, target = batch[1].to(dev), batch[2].to(dev), batch[3].to(dev), batch[4].to(dev)

        with torch.no_grad():
            pred = model(leye, reye, kps)

        preds.extend(pred.cpu().numpy())
        gt.extend(target.cpu().numpy())

    preds = np.array(preds)
    gt = np.array(gt)

    pts = np.unique(gt, axis=0)
    c = get_colors(len(pts))
    random.shuffle(c)

    dist = euc(preds, gt)
    print("Mean Euclidean Distance:", dist.mean())

if __name__ == "__main__":
    main()
