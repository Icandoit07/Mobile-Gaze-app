import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

class SVRPersonalization:
    def __init__(self, eye_tracker_model, device='cpu'):
        self.eye_tracker_model = eye_tracker_model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    def extract_features(self, dataloader):
        self.eye_tracker_model.eval()
        features = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                leye, reye, kps, target = (
                    batch["left"].to(self.device),
                    batch["right"].to(self.device),
                    batch["face"].to(self.device),
                    batch["grid"].to(self.device),
                    batch["gaze"].to(self.device)
                )

                input_data = {"left": leye, "right": reye, "face": kps, "grid": target}
                feature = self.eye_tracker_model(input_data)
                
                features.append(feature.cpu().numpy())
                targets.append(target.cpu().numpy())

        return np.vstack(features), np.vstack(targets)

    def train(self, dataloader):
        features, targets = self.extract_features(dataloader)
 
        features = self.scaler.fit_transform(features)


        X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)


        self.svr_model.fit(X_train, y_train)


        score = self.svr_model.score(X_val, y_val)
        print(f"SVR Validation R^2 Score: {score:.4f}")

    def predict(self, dataloader):
        features, targets = self.extract_features(dataloader)
        

        features = self.scaler.transform(features)
        

        predictions = self.svr_model.predict(features)

      
        error = np.mean(np.linalg.norm(predictions - targets, axis=1))
        print(f"Mean Euclidean Distance Error: {error:.4f}")

        return predictions

if __name__ == "__main__":

    from dataset import MITGazeCaptureDataset
    from model import EyeTrackerModel


    eye_tracker_model = EyeTrackerModel()


    user_id = 'your_user_id'
    dataset = MITGazeCaptureDataset(user_id=user_id, phase='train')  # Customize dataset loading based on user ID
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    svr_personalizer = SVRPersonalization(eye_tracker_model, device='cuda' if torch.cuda.is_available() else 'cpu')


    svr_personalizer.train(dataloader)

    test_dataset = MITGazeCaptureDataset(user_id=user_id, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    svr_personalizer.predict(test_dataloader)
