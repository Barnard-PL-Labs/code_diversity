import os
import numpy as np
from wavEmbed import embed_wav
from codeEmbed import model as code_model
import torch
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def get_dataset_pairs():
    """Get all code-wav pairs from the dataset folder"""
    pairs = []
    dataset_path = "dataset"
    
    for example_name in os.listdir(dataset_path):
        example_dir = os.path.join(dataset_path, example_name)
        if os.path.isdir(example_dir):
            # Find the code file (assuming it ends with .pi)
            code_file = next((f for f in os.listdir(example_dir) if f.endswith('.pi')), None)
            # Find the wav file
            wav_file = next((f for f in os.listdir(example_dir) if f.endswith('.wav')), None)
            
            if code_file and wav_file:
                pairs.append({
                    'name': example_name,
                    'code_path': os.path.join(example_dir, code_file),
                    'wav_path': os.path.join(example_dir, wav_file)
                })
    
    return pairs

def calculate_distance_correlations():
    pairs = get_dataset_pairs()
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
    
    for pair1, pair2 in sample_pairs:
        name1, name2 = pair1['name'], pair2['name']
        
        # Calculate distances in code space
        code_dist = torch.dist(
            code_embeddings[name1],
            code_embeddings[name2]
        ).item()
        
        # Calculate distances in wav space
        wav_dist = torch.dist(
            wav_embeddings[name1],
            wav_embeddings[name2]
        ).item()
        
        results.append({
            'pair': f"{name1} vs {name2}",
            'code_distance': code_dist,
            'wav_distance': wav_dist
        })
    
    df = pd.DataFrame(results)
    
    # Calculate correlation between distances
    correlation_pearson, p_value_pearson = pearsonr(
        df['code_distance'], 
        df['wav_distance']
    )
    
    correlation_spearman, p_value_spearman = spearmanr(
        df['code_distance'], 
        df['wav_distance']
    )
    
    print("\nCorrelation Analysis:")
    print(f"Pearson correlation: {correlation_pearson:.4f} (p={p_value_pearson:.4f})")
    print(f"Spearman correlation: {correlation_spearman:.4f} (p={p_value_spearman:.4f})")
    
    # Save results
    df.to_csv('distance_analysis.csv', index=False)
    return df, correlation_pearson, correlation_spearman

def plot_distance_correlation(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['code_distance'], df['wav_distance'])
    plt.xlabel('Distance in Code Space')
    plt.ylabel('Distance in WAV Space')
    plt.title('Relationship between Code and WAV Distances')
    plt.savefig('distance_correlation.png')
    plt.close()

if __name__ == "__main__":
    df, pearson, spearman = calculate_distance_correlations()
    print("\nPairwise Distances:")
    print(df)
    
    plot_distance_correlation(df)
    
    # this should grab all the files in the dataset folder and then embed the code and embed the wav
    # then calculate the pairwise distances between the code and the wav embedding
    # we will use this to somehow show that there does not exist a linear mapping from  the code to wave embeddings 
