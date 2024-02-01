import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import model
from gazetrack_dataset import gazetrack_dataset

def get_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors

def euc(preds, gt):
    return np.linalg.norm(preds - gt, axis=1)

def plot_predictions(gt, preds, title='Predictions'):
    plt.figure(figsize=(8, 6))
    plt.scatter(gt[:, 0], gt[:, 1], color='blue', label='Ground Truth')
    plt.scatter(preds[:, 0], preds[:, 1], color='red', label='Predictions')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

def svr_personalization(features, targets):

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(features_scaled, targets)


    predictions = svr_model.predict(features_scaled)
    return predictions, svr_model

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds, gt = [], []

    # Load test data
    f = files[idx]
    fs = glob.glob(f + "*.jpg")
    test_dataset = gazetrack_dataset(f, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=False, shuffle=False)

    eye_tracker_model = model.EyeTrackerModel().to(dev) 

    for batch in tqdm(test_dataloader):
        leye, reye, kps, target = batch[1].to(dev), batch[2].to(dev), batch[3].to(dev), batch[4].to(dev)

        with torch.no_grad():
            pred = eye_tracker_model(leye, reye, kps)

        preds.extend(pred.cpu().numpy())
        gt.extend(target.cpu().numpy())

    preds = np.array(preds)
    gt = np.array(gt)

    plot_predictions(gt, preds, title='Initial Predictions')


    features = preds 
    targets = gt   

    svr_preds, svr_model = svr_personalization(features, targets)


    plot_predictions(gt, svr_preds, title='SVR Personalized Predictions')

    initial_dist = euc(preds, gt)
    svr_dist = euc(svr_preds, gt)
    
    print("Mean Euclidean Distance (Initial):", initial_dist.mean())
    print("Mean Euclidean Distance (SVR Personalized):", svr_dist.mean())

if __name__ == "__main__":
    main()
