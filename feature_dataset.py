import pandas as pd
from torch.utils.data import Dataset
import torch

class CustomFeatureDataset(Dataset):

    def __init__(self, pkl_file):

        self.df = pd.read_pickle(pkl_file)

        self.feature_vectors = self.df['feature_vector'].tolist()
        self.video_paths = self.df['video_path'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        feature_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32)
        video_path = self.video_paths[idx]
        label = self.labels[idx]


        return feature_vector, video_path, label
    

    

