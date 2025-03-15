#!/usr/bin/env python
# coding: utf-8

# ## **Checking**

# In[ ]:


import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(" GPU Not Found, Using CPU!")


# In[ ]:


# Google Drive import
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')


# In[ ]:


# Define dataset path
data_path = "/content/drive/My Drive/ColabNotebooks/nuScenes"

# Check if Train, Validation, and Test folders exist
folders = ["train", "val", "test"]
for folder in folders:
    full_path = os.path.join(data_path, folder)
    if os.path.exists(full_path):
        print(f"{folder} folder found: {full_path}")
    else:
        print(f"{folder} folder not found!")


# ## **Import Libraries**

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random


# ## **Set a fixed random seed**

# In[ ]:


# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# If using GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED}")


# ## **NuScenesDataset Loader**

# In[ ]:


# This dataset loader assumes that each CSV file in your provided directory
# Adjust the processing if your CSV format differs.

class NuScenesDataset(Dataset):
    def __init__(self, data_dir, split="train", history_frames=4, future_frames=6):

        """
        Args:
            data_dir (str): Directory containing CSV files.
            split (str): "train" or "val".
            history_frames (int): Number of history frames.
            future_frames (int): Number of future frames to predict.
        """

        self.data_dir = data_dir
        self.split = split
        self.history_frames = history_frames
        self.future_frames = future_frames

        # Load all CSV files from the directory
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        if len(self.files) == 0:
            raise ValueError(f"No CSV files found in directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        """
        Loads and processes a single CSV file to extract agent trajectories.

        Args:
        - idx (int): Index of the CSV file.

        Returns:
        - sample (dict): Processed data containing agent features, motifs, adjacency matrix, and ground truth.
        """

        # Read the CSV file
        csv_file = self.files[idx]
        df = pd.read_csv(csv_file)

        # Extract unique agents and timestamps
        agents = df['agent_id'].unique()
        frames = sorted(df['frame_id'].unique())

        # Ensure the CSV has enough frames for history and prediction
        if len(frames) < (self.history_frames + self.future_frames + 1):
            raise ValueError(f"Not enough frames in CSV: {csv_file}")


        random.seed(SEED)  # Ensures consistent selection of timestamps
        np.random.seed(SEED)

        T_idx = random.randint(self.history_frames, len(frames) - self.future_frames - 1)
        T = frames[T_idx]

        agent_features_list = []
        gt_list = []
        positions_T = {}

        # Process each agent's trajectory
        for agent in agents:
            agent_df = df[df['agent_id'] == agent].sort_values(by='frame_id')

            # Ensure agent has enough frames
            if len(agent_df) < (self.history_frames + self.future_frames):
                continue
            row_T = agent_df[agent_df['frame_id'] == T]

            # Extract the row corresponding to time T
            if row_T.empty:
                continue
            row_T = row_T.iloc[0]

            # Agent features: at time T
            agent_feat = np.array([row_T['x'], row_T['y'], row_T['heading'], row_T['speed']], dtype=np.float32)
            agent_features_list.append(agent_feat)
            positions_T[agent] = np.array([row_T['x'], row_T['y']], dtype=np.float32)

            # Ground truth: future positions from T+1 to T+future_frames (x, y only)
            future_positions = []
            for future_fid in frames[T_idx+1 : T_idx+1+self.future_frames]:
                row_future = agent_df[agent_df['frame_id'] == future_fid]
                if row_future.empty:
                    future_positions.append(agent_feat[:2]) # Use last known position if missing
                else:
                    row_future = row_future.iloc[0]
                    future_positions.append(np.array([row_future['x'], row_future['y']], dtype=np.float32))
            gt_list.append(np.stack(future_positions))

        # Ensure we have valid agents
        if len(agent_features_list) == 0:
            raise ValueError(f"No agents with enough frames in CSV: {csv_file}")

        # Convert lists to numpy arrays
        agent_features = np.stack(agent_features_list)  # shape: (N, 4)
        gt = np.stack(gt_list)                          # shape: (N, future_frames, 2)
        num_agents = agent_features.shape[0]

        # Compute adjacency matrix based on distance at time T
        threshold = 10.0  # meters
        adj = np.zeros((num_agents, num_agents), dtype=np.float32)
        positions = np.array([agent_features[i][:2] for i in range(num_agents)], dtype=np.float32)
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    adj[i, j] = 1.0 # Self-connection
                else:
                    if np.linalg.norm(positions[i] - positions[j]) < threshold:
                        adj[i, j] = 1.0

        # Compute spatial motif 
        spatial_motif_list = []
        for i in range(num_agents):
            dists = []
            for j in range(num_agents):
                if i == j:
                    continue
                dists.append(np.linalg.norm(positions[i] - positions[j]))
            dists = sorted(dists)
            motif_vec = np.zeros(16, dtype=np.float32)
            length = min(len(dists), 16)
            motif_vec[:length] = dists[:length]
            spatial_motif_list.append(motif_vec)
        spatial_motif = np.stack(spatial_motif_list)  # shape: (N, 16)

        # Compute temporal motif: for each agent, difference between future positions and current 
        temporal_motif_list = []
        for i in range(num_agents):
            curr_pos = agent_features[i][:2]
            diffs = []
            for t in range(self.future_frames):
                diff = gt[i][t] - curr_pos # Difference between future and current positions
                temp_vec = np.zeros(16, dtype=np.float32)
                temp_vec[:2] = diff # Store x, y difference
                diffs.append(temp_vec)
            diffs = np.stack(diffs)  # (future_frames, 16)
            temporal_motif_list.append(diffs)
        temporal_motif = np.stack(temporal_motif_list)  # shape: (N, future_frames, 16)

        # Convert to Torch Tensors
        sample = {
            "agent_features": torch.tensor(agent_features, dtype=torch.float32),
            "spatial_motif": torch.tensor(spatial_motif, dtype=torch.float32),
            "temporal_motif": torch.tensor(temporal_motif, dtype=torch.float32),
            "adj": torch.tensor(adj, dtype=torch.float32),
            "gt": torch.tensor(gt, dtype=torch.float32)
        }
        return sample


# ## **Model Variants (Baseline and DAMM Variants)**

# In[ ]:


# M1: Baseline (no STMM, no ASI, no ATI)
class BaselineModel(nn.Module):
    def __init__(self, agent_input_dim=4, future_frames=6):

        """
        Initializes the baseline model.

        Args:
        - agent_input_dim (int): Number of input features per agent (x, y, speed, heading).
        - future_frames (int): Number of future time steps to predict.
        """

        super().__init__()
        hidden_dim = 32

        # Simple MLP for trajectory prediction
        self.mlp = nn.Sequential(
            nn.Linear(agent_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_frames * 2) # Predicts (x, y) for each future frame
        )
        self.future_frames = future_frames

    def forward(self, agent_features, spatial_motif, temporal_motif, adj):

        B, N, _ = agent_features.shape

        # Flatten agent features and pass through MLP
        preds = self.mlp(agent_features.view(-1, agent_features.shape[-1])) # (B * N, future_frames * 2)

        # Reshape predictions back to (B, N, future_steps, 2)
        preds = preds.view(B, N, self.future_frames, 2)

        return preds


# In[ ]:


# DAMMVariant: A flexible model that can toggle usage of STMM, ASI, ATI.
# M2: STMM + ASI (use_asi=True, use_ati=False)
# M3: STMM + ATI (use_asi=False, use_ati=True)
# M4: Full model (use_asi=True, use_ati=True)

class DAMMVariant(nn.Module):
    def __init__(self, use_stmm=True, use_asi=True, use_ati=True,
                 agent_input_dim=4, motif_spatial_dim=16, motif_temporal_dim=16,
                 future_frames=6):

        """
        Initializes a flexible DAMM variant that can selectively enable or disable STMM, ASI, and ATI.

        Args:
        - use_stmm (bool): Whether to use Spatial-Temporal Motif Matrix (STMM).
        - use_asi (bool): Whether to use Adaptive Spatial Interaction (ASI).
        - use_ati (bool): Whether to use Adaptive Temporal Interaction (ATI).
        - agent_input_dim (int): Input feature size.
        - motif_spatial_dim (int): Dimension of spatial motifs.
        - motif_temporal_dim (int): Dimension of temporal motifs.
        - future_frames (int): Number of future steps to predict.
        """

        super().__init__()
        self.use_stmm = use_stmm
        self.use_asi = use_asi
        self.use_ati = use_ati
        self.future_frames = future_frames

        hidden_dim = 32

        # Base agent encoder
        self.agent_fc = nn.Sequential(
            nn.Linear(agent_input_dim, hidden_dim),
            nn.ReLU()
        )
        if self.use_stmm:
            self.spatial_fc = nn.Sequential(
                nn.Linear(motif_spatial_dim, hidden_dim),
                nn.ReLU()
            )

            # Temporal feature encoder (ATI)
            self.temporal_fc = nn.Sequential(
                nn.Linear(motif_temporal_dim, hidden_dim),
                nn.ReLU()
            )

        # Define the final combined feature size
        combine_in_dim = hidden_dim
        if self.use_stmm:
            if self.use_asi:
                combine_in_dim += hidden_dim # Adding ASI features
            if self.use_ati:
                combine_in_dim += hidden_dim # Adding ATI features

        # Fully connected layer to combine features
        self.combine_fc = nn.Sequential(
            nn.Linear(combine_in_dim, hidden_dim),
            nn.ReLU()
        )

        # Output layer predicting (x, y) for each future time step
        self.out_fc = nn.Linear(hidden_dim, future_frames * 2)

    def forward(self, agent_features, spatial_motif, temporal_motif, adj):

        B, N, _ = agent_features.shape
        outputs = []

        for b in range(B):

            # Encode agent-specific features
            agent_enc = self.agent_fc(agent_features[b])

            # Initialize feature list
            fused = [agent_enc]

            if self.use_stmm:
                if self.use_asi:
                    spatial_enc = self.spatial_fc(spatial_motif[b])
                    fused.append(spatial_enc)

                if self.use_ati:
                    # Temporal motif processing: Average across time dimension
                    temporal_avg = torch.mean(temporal_motif[b], dim=1)
                    temporal_enc = self.temporal_fc(temporal_avg)
                    fused.append(temporal_enc)

            # Concatenate selected features
            combined = torch.cat(fused, dim=-1)
            combined = self.combine_fc(combined) # Pass through final FC layer

            # Generate predictions
            preds = self.out_fc(combined) # (N, future_frames * 2)
            preds = preds.view(N, self.future_frames, 2)

            outputs.append(preds)

        return torch.stack(outputs, dim=0) # (B, N, future_frames, 2)


# ## **Loss and Metrics (ADE and FDE)**

# In[ ]:


def compute_loss(pred, gt):

    """
    Computes Mean Squared Error (MSE) loss between predictions and ground truth.

    Args:
    - pred (Tensor): Predicted future trajectory (B, N, T, 2).
    - gt (Tensor): Ground truth trajectory (B, N, T, 2).

    Returns:
    - Tensor: MSE loss value.
    """

    mse_loss = nn.MSELoss()
    return mse_loss(pred, gt)


# In[ ]:


def compute_ade_fde(pred, gt):

    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Args:
    - pred (Tensor): Predicted trajectory (B, N, T, 2).
    - gt (Tensor): Ground truth trajectory (B, N, T, 2).

    Returns:
    - ade (float): Average Displacement Error.
    - fde (float): Final Displacement Error.
    """

    # Compute displacement differences
    diff = pred - gt # (B, N, T, 2)

    # Compute Euclidean distances at each time step
    dist = torch.norm(diff, dim=-1) # (B, N, T)

    # ADE: Average over all time steps
    ade = torch.mean(dist)

    # FDE: Distance at final time step (T)
    fde = torch.mean(dist[:, :, -1]) # Final displacement

    return ade.item(), fde.item()


# In[ ]:


def compute_ade_fde_topk(model, batch, K=1):

    """
    Computes ADE and FDE for top-K predictions by sampling multiple outputs.

    Args:
    - model: Trained trajectory prediction model.
    - batch: Input batch containing agent features, spatial/temporal motifs, adjacency matrix, and ground truth.
    - K (int): Number of sampled trajectory predictions.

    Returns:
    - ade (float): Best ADE among the K predictions.
    - fde (float): Best FDE among the K predictions.
    """

    # Expand input batch for K samples
    agent_features = batch["agent_features"].unsqueeze(0)  # (1, N, 4)
    spatial_motif  = batch["spatial_motif"].unsqueeze(0)   # (1, N, 16)
    temporal_motif = batch["temporal_motif"].unsqueeze(0)  # (1, N, T, 16)
    adj = batch["adj"].unsqueeze(0)                        # (1, N, N)
    gt = batch["gt"].unsqueeze(0)                          # (1, N, T, 2)

    preds_list = []

    # Generate K different trajectory predictions
    for _ in range(K):
        pred = model(agent_features, spatial_motif, temporal_motif, adj) # (1, N, T, 2)
        preds_list.append(pred)

    # Convert predictions to tensor shape
    all_preds = torch.cat(preds_list, dim=0)  # (K, 1, N, T, 2)

    # Compute displacement distances
    dist = torch.norm(all_preds - gt, dim=-1)  # (K, 1, N, T)

    # Select the best prediction per agent (minimum error)
    min_dist, _ = torch.min(dist, dim=0)       # (1, N, T)

    # Compute best ADE and FDE
    ade = torch.mean(min_dist)
    fde = torch.mean(min_dist[:, :, -1]) # Final step error

    return ade.item(), fde.item()


# ## **Training and Ablation Experiment**

# In[ ]:


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0 # Set model to training mode

    torch.manual_seed(SEED)  # Ensures deterministic training behavior

    for batch in dataloader:

        # Add batch dimension if not present
        agent_features = batch["agent_features"].to(device).unsqueeze(0)
        spatial_motif  = batch["spatial_motif"].to(device).unsqueeze(0)
        temporal_motif = batch["temporal_motif"].to(device).unsqueeze(0)
        adj = batch["adj"].to(device).unsqueeze(0)
        gt  = batch["gt"].to(device).unsqueeze(0)

        # Forward pass
        optimizer.zero_grad()
        pred = model(agent_features, spatial_motif, temporal_motif, adj)
        loss = compute_loss(pred, gt)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# In[ ]:


# Function to evaluate the model
def evaluate_model(model, dataloader, device, K_list=[1,5,10]):
    model.eval()
    results = {f"ADE{k}": [] for k in K_list}
    results.update({f"FDE{k}": [] for k in K_list})

    with torch.no_grad(): # Disable gradient computation
        for batch in dataloader:
            for k in K_list:
                ade, fde = compute_ade_fde_topk(model, batch, K=k)
                results[f"ADE{k}"].append(ade)
                results[f"FDE{k}"].append(fde)

    # Compute average values for ADE and FDE
    avg = {key: float(np.mean(val)) for key, val in results.items()}
    return avg


# In[ ]:


# Function to run the ablation experiment
def run_ablation_experiment(train_data_dir, val_data_dir, device, epochs=5):

    torch.manual_seed(SEED)  # Ensures deterministic model weight initialization


    # Load datasets
    train_dataset = NuScenesDataset(train_data_dir, split="train")
    val_dataset   = NuScenesDataset(val_data_dir, split="val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Define model variants with corresponding flags:
    # M1: Baseline (no STMM, ATI, ASI)
    M1_model = BaselineModel()
    M1_model.to(device)
    M1_opt = optim.Adam(M1_model.parameters(), lr=1e-3)

    # M2: STMM + ASI, no ATI
    M2_model = DAMMVariant(use_stmm=True, use_asi=True, use_ati=False)
    M2_model.to(device)
    M2_opt = optim.Adam(M2_model.parameters(), lr=1e-3)

    # M3: STMM + ATI, no ASI
    M3_model = DAMMVariant(use_stmm=True, use_asi=False, use_ati=True)
    M3_model.to(device)
    M3_opt = optim.Adam(M3_model.parameters(), lr=1e-3)

    # M4: STMM + ASI + ATI
    M4_model = DAMMVariant(use_stmm=True, use_asi=True, use_ati=True)
    M4_model.to(device)
    M4_opt = optim.Adam(M4_model.parameters(), lr=1e-3)

    # Store configuration for printing:
    variants = {
        "M1": {"model": M1_model, "STMM": "×", "ATI": "×", "ASI": "×"},
        "M2": {"model": M2_model, "STMM": "√", "ATI": "×", "ASI": "√"},
        "M3": {"model": M3_model, "STMM": "√", "ATI": "√", "ASI": "×"},
        "M4": {"model": M4_model, "STMM": "√", "ATI": "√", "ASI": "√"}
    }

    # Train
    for epoch in range(epochs):
        for key, variant in variants.items():
            train_epoch(variant["model"], train_loader, optim.Adam(variant["model"].parameters(), lr=1e-3), device)

    # Evaluate each variant on the validation set for K=1,5,10
    metrics = {}
    for key, variant in variants.items():
        metrics[key] = evaluate_model(variant["model"], val_loader, device, K_list=[1,5,10])

    # Function to format ADE/FDE as string "ade/fde"
    def fmt_metric(m, k):
        return f"{m[f'ADE{k}']:.2f}/{m[f'FDE{k}']:.2f}"

    def improvement(baseline_str, variant_str):
        ade_base, fde_base = map(float, baseline_str.split("/"))
        ade_var, fde_var = map(float, variant_str.split("/"))
        imp_ade = (ade_base - ade_var) / ade_base * 100 if ade_base != 0 else 0
        imp_fde = (fde_base - fde_var) / fde_base * 100 if fde_base != 0 else 0
        return f"{imp_ade:.2f}%/{imp_fde:.2f}%"

    # Print the table header:
    print(" Ablation study conducted on nuScenes".center(100, "="))
    header = ("M*", "STMM", "ATI", "ASI",
              "ADE1/FDE1", "Percentage", "ADE5/FDE5", "Percentage", "ADE10/FDE10", "Percentage")
    print("{:<7} {:<6} {:<6} {:<6} {:<12} {:<15} {:<12} {:<15} {:<15} {:<15}".format(*header))
    print("-"*100)

    # Baseline M1 formatted metrics
    M1_metrics = {k: fmt_metric(metrics["M1"], k) for k in [1,5,10]}

    # Print performance metrics for each model
    for key in ["M1", "M2", "M3", "M4"]:
        variant = variants[key]
        m_str_1 = fmt_metric(metrics[key], 1)
        m_str_5 = fmt_metric(metrics[key], 5)
        m_str_10 = fmt_metric(metrics[key], 10)
        if key == "M1":
            imp1 = imp5 = imp10 = "-"
        else:
            imp1 = improvement(M1_metrics[1], m_str_1)
            imp5 = improvement(M1_metrics[5], m_str_5)
            imp10 = improvement(M1_metrics[10], m_str_10)
        print("{:<7} {:<6} {:<6} {:<6} {:<12} {:<15} {:<12} {:<15} {:<15} {:<15}".format(
            key, variant["STMM"], variant["ATI"], variant["ASI"], m_str_1, imp1, m_str_5, imp5, m_str_10, imp10
        ))

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset directories
    train_data_dir = "/content/drive/My Drive/ColabNotebooks/nuScenes/train/data"
    val_data_dir = "/content/drive/My Drive/ColabNotebooks/nuScenes/val/data"

    # Run the ablation experiment
    run_ablation_experiment(train_data_dir, val_data_dir, device, epochs=5)

if __name__ == "__main__":
    main()


# ---

# ***Zahra Amanli..***
