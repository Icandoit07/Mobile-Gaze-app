import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import torch

def gaze_to_2d(gaze: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])

class GazeDataset(Dataset):
    def __init__(self, paths: List[str], root: str, header: bool = True):
        self.root = root
        self.lines = self._load_files(paths, header)

    def _load_files(self, paths: List[str], header: bool) -> List[str]:
        lines = []
        for path in paths:
            with open(path, 'r') as f:
                file_lines = f.readlines()
                if header:
                    file_lines.pop(0)
                lines.extend(file_lines)
        return lines

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        line_data = self.lines[idx].strip().split()
        name, lefteye, righteye, grid, point, _, device = line_data[:7]

        label = torch.tensor([float(x) for x in point.split(',')], dtype=torch.float32)

        img_data = {
            "left": self._load_image(lefteye),
            "right": self._load_image(righteye),
            "face": self._load_image(name),
            "grid": self._load_grid(grid),
            "name": name,
            "device": device
        }

        return img_data, label

    def _load_image(self, filename: str) -> torch.Tensor:
        img = cv2.imread(os.path.join(self.root, filename))
        img = cv2.resize(img, (224, 224)) / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def _load_grid(self, filename: str) -> torch.Tensor:
        grid = cv2.imread(os.path.join(self.root, filename), 0)
        grid = np.expand_dims(grid, 0)
        return torch.from_numpy(grid).float()

def load_gaze_data(label_paths: List[str], image_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0, header: bool = True) -> DataLoader:
    dataset = GazeDataset(label_paths, image_path, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    # Using dummy paths
    label_dir = "/path/to/label/directory"
    image_dir = "/path/to/image/directory"
    
    # Assume we have some label files in the directory
    label_files = [os.path.join(label_dir, f"label_{i}.txt") for i in range(5)]
    
    data_loader = load_gaze_data(label_files, image_dir, batch_size=10)
    
    print(f"Number of batches: {len(data_loader)}")
    
    # Example of iterating through the data
    for batch_data, batch_labels in data_loader:
        # Process your batch here
        print(f"Batch shape: {batch_data['left'].shape}")
        print(f"Label shape: {batch_labels.shape}")
        break  # Just process one batch for this example