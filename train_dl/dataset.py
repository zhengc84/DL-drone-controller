import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DroneSequenceDataset(Dataset):
    def __init__(self, file_path, sequence_length=50):
        print(f"Loading data from {file_path}...")
        
        # 1. Load the raw data
        if file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        else:
            self.df = pd.read_csv(file_path)
            
        # --- 2. THE GLITCH FIX ---
        # Delete the very first row (index 0) of every episode 
        # to kill the 1000-point velocity spike artifact
        self.df = self.df[self.df.groupby('episode').cumcount() > 0].reset_index(drop=True)
        # -------------------------
            
        self.sequence_length = sequence_length
        self.feature_cols = ['error', 'error_dot', 'integral', 'prev_thrust']
        self.label_cols = ['target_kp', 'target_ki', 'target_kd']
        
        # --- 3. THE NORMALIZATION SHIELD ---
        # Calculate mean and std dev, then scale all inputs to ~ [-1.0, 1.0]
        raw_features = self.df[self.feature_cols].values
        self.mean = raw_features.mean(axis=0)
        self.std = raw_features.std(axis=0) + 1e-8
        self.df[self.feature_cols] = (raw_features - self.mean) / self.std
        # -----------------------------------
        
        self.X = []
        self.Y = []
        self._build_sequences()

    def _build_sequences(self):
        grouped_episodes = self.df.groupby('episode')
        
        for episode_id, group in grouped_episodes:
            features = group[self.feature_cols].values
            labels = group[self.label_cols].values
            
            # Slide a 50-step window across the flight data
            for i in range(len(group) - self.sequence_length):
                seq_x = features[i : i + self.sequence_length]
                seq_y = labels[i + self.sequence_length] 
                
                self.X.append(seq_x)
                self.Y.append(seq_y)
                
        print(f"Created {len(self.X)} pristine, glitch-free sequence windows.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x_tensor, y_tensor