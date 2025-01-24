import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from wavEmbed import embed_wav
from codeEmbed import model as code_model
from itertools import combinations
from main import get_dataset_pairs

class CodeAudioDistanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DistancePredictor(nn.Module):
    def __init__(self, input_dim=1536, hidden1=512, hidden2=256):
        super(DistancePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.model(x)

def shuffle_split(dataset, N):
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    return shuffled[:N], shuffled[N:]

def process_dataset_pairs(pairs):
    code_embeddings = {}
    wav_embeddings = {}

    # First, compute all embeddings
    for pair in pairs:
        with open(pair['code_path'], 'r') as f:
            code_content = f.read()
        code_embeddings[pair['name']] = code_model.encode(code_content, convert_to_tensor=True)
        wav_embeddings[pair['name']] = embed_wav(pair['wav_path']).squeeze()

    # Compute pairwise distances within each space
    sample_pairs = list(combinations(pairs, 2))
    results = []
    
    for pair in sample_pairs:
        name1, name2 = pair[0]['name'], pair[1]['name']
        code_dist = torch.dist(code_embeddings[name1], code_embeddings[name2]).item()
        wav_dist = torch.dist(wav_embeddings[name1], wav_embeddings[name2]).item()
        
        results.append({
            'a': name1,
            'code_embedding_a': code_embeddings[name1],
            'b': name2,
            'code_embedding_b': code_embeddings[name2],
            'code_distance': code_dist,
            'wav_distance': wav_dist
        })

    return pd.DataFrame(results)

def torch_input_targets(df):
    inputs = []
    targets = []

    for idx, row in df.iterrows():
        emb_a = row["code_embedding_a"]
        emb_b = row["code_embedding_b"]

        # Convert to torch tensors if not already
        emb_a = torch.tensor(emb_a, dtype=torch.float32) if not isinstance(emb_a, torch.Tensor) else emb_a.float()
        emb_b = torch.tensor(emb_b, dtype=torch.float32) if not isinstance(emb_b, torch.Tensor) else emb_b.float()
        emb_a = emb_a.to("cpu")
        emb_b = emb_b.to("cpu")

        concat_emb = torch.cat([emb_a, emb_b], dim=0)
        wav_dist = float(row["wav_distance"])

        inputs.append(concat_emb)
        targets.append(wav_dist)

    inputs = torch.stack(inputs)  # shape: (N, 1536)
    targets = torch.tensor(targets).float().view(-1, 1)  # shape: (N, 1)
    return inputs, targets

def train_model(model, train_loader, test_loader, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                preds_test = model(batch_X_test)
                loss_test = criterion(preds_test, batch_y_test)
                test_loss += loss_test.item() * batch_X_test.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            preds_test = model(batch_X_test)
            all_preds.append(preds_test.cpu())
            all_targets.append(batch_y_test.cpu())

    all_preds = torch.cat(all_preds).squeeze()
    all_targets = torch.cat(all_targets).squeeze()

    mse_val = mean_squared_error(all_targets, all_preds)
    pearson_corr, _ = pearsonr(all_targets, all_preds)
    spearman_corr, _ = spearmanr(all_targets, all_preds)

    print(f"Test MSE: {mse_val:.4f}")
    print(f"Pearson Corr: {pearson_corr:.4f}")
    print(f"Spearman Corr: {spearman_corr:.4f}")

def main():
    # Get all subdirectories in dataset_vari
    variants = [d for d in os.listdir("dataset_vari") 
               if os.path.isdir(os.path.join("dataset_vari", d))]

    # Initialize and populate dataset pairs
    init = get_dataset_pairs()
    for variant in variants:
        dataset_path = f"dataset_vari/{variant}"
        init += get_dataset_pairs(dataset_path)

    # Split into train and test sets
    test_pairs, pairs = shuffle_split(init, 8)
    print("Test Pairs:", test_pairs)

    # Process datasets
    df = process_dataset_pairs(pairs)
    test_df = process_dataset_pairs(test_pairs)

    # Prepare input tensors
    inputs, targets = torch_input_targets(df)
    test_inputs, test_targets = torch_input_targets(test_df)
    print("TrainInputs Shape:", inputs.shape)
    print("TestInputs Shape:", test_inputs.shape)

    # Create datasets and dataloaders
    train_dataset = CodeAudioDistanceDataset(inputs, targets)
    test_dataset = CodeAudioDistanceDataset(test_inputs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize and train model
    model = DistancePredictor(input_dim=1536)
    train_model(model, train_loader, test_loader)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
