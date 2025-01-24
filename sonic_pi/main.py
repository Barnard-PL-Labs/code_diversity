import os
import numpy as np
from wavEmbed import embed_wav
from codeEmbed import model as code_model
import torch
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import argparse

def get_dataset_pairs(dataset_path="dataset", max_samples=None):
    """Get all code-wav pairs from the dataset folder
    
    Args:
        dataset_path (str): Path to the dataset folder. Defaults to "dataset".
        max_samples (int, optional): Maximum number of pairs to return. If None, return all pairs.
    """
    pairs = []
    
    for example_name in os.listdir(dataset_path):
        # Break if we've reached max_samples
        if max_samples and len(pairs) >= max_samples:
            break
            
        example_dir = os.path.join(dataset_path, example_name)
        if os.path.isdir(example_dir):
            # Find the code file (assuming it ends with .pi)
            code_file = next((f for f in os.listdir(example_dir) if f.endswith('.pi') or f.endswith('.rb')), None)
            # Find the wav file
            wav_file = next((f for f in os.listdir(example_dir) if f.endswith('.wav')), None)
            
            if code_file and wav_file:
                pairs.append({
                    'name': example_name,
                    'code_path': os.path.join(example_dir, code_file),
                    'wav_path': os.path.join(example_dir, wav_file)
                })
    
    return pairs

def generate_correlation_data():
    """Generate and save the correlation analysis data"""
    pairs = get_dataset_pairs()
    code_embeddings = {}
    wav_embeddings = {}
    
    # First, compute all embeddings
    for pair in pairs:
        with open(pair['code_path'], 'r') as f:
            code_content = f.read()
        code_embeddings[pair['name']] = code_model.encode(code_content, convert_to_tensor=True)
        wav_embeddings[pair['name']] = embed_wav(pair['wav_path']).squeeze()
    
    # Compute pairwise similarities within each space
    sample_pairs = list(combinations(pairs, 2))
    results = []
    
    for pair1, pair2 in sample_pairs:
        name1, name2 = pair1['name'], pair2['name']
        
        # Calculate cosine similarity in code space
        code_sim = torch.nn.functional.cosine_similarity(
            code_embeddings[name1].unsqueeze(0),
            code_embeddings[name2].unsqueeze(0)
        ).item()
        
        # Calculate cosine similarity in wav space
        wav_sim = torch.nn.functional.cosine_similarity(
            wav_embeddings[name1].unsqueeze(0),
            wav_embeddings[name2].unsqueeze(0)
        ).item()
        
        results.append({
            'pair': f"{name1} vs {name2}",
            'code_similarity': code_sim,
            'wav_similarity': wav_sim
        })
    
    df = pd.DataFrame(results)
    
    # Calculate correlation between similarities
    correlation_pearson, p_value_pearson = pearsonr(
        df['code_similarity'], 
        df['wav_similarity']
    )
    
    correlation_spearman, p_value_spearman = spearmanr(
        df['code_similarity'], 
        df['wav_similarity']
    )
    
    print("\nCorrelation Analysis:")
    print(f"Pearson correlation: {correlation_pearson:.4f} (p={p_value_pearson:.4f})")
    print(f"Spearman correlation: {correlation_spearman:.4f} (p={p_value_spearman:.4f})")
    
    # Save results
    df.to_csv('similarity_analysis.csv', index=False)
    return df, correlation_pearson, correlation_spearman

def plot_correlation_analysis(df=None):
    """Generate plots from correlation analysis data"""
    if df is None:
        # Load data if not provided
        df = pd.read_csv('similarity_analysis.csv')
    
    # Set up Computer Modern font with bold weight
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.weight'] = 'bold'
    
    # Create figure with larger size
    plt.figure(figsize=(12, 12))
    
    # Base font size for consistency
    base_fontsize = 40
    
    # Determine the global min and max for both axes
    min_val = min(df['code_similarity'].min(), df['wav_similarity'].min())
    max_val = max(df['code_similarity'].max(), df['wav_similarity'].max())
    
    # Set axis limits before plotting points
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.scatter(df['code_similarity'], df['wav_similarity'], s=130, alpha=0.6)
    
    # Increase font sizes using base_fontsize
    plt.xlabel('Cosine Similarity in Code Space', fontsize=base_fontsize, fontweight='bold')
    plt.ylabel('Cosine Similarity in WAV Space', fontsize=base_fontsize, fontweight='bold')
    plt.title('Code and WAV Similarities on Examples', fontsize=base_fontsize, fontweight='bold')
    
    # Increase tick label sizes
    plt.xticks(fontsize=base_fontsize, fontweight='bold')
    plt.yticks(fontsize=base_fontsize, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('similarity_correlation.pdf', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate new analysis data')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    if args.generate:
        df, pearson, spearman = generate_correlation_data()
        print("\nPairwise Distances:")
        print(df)
        if args.plot:
            plot_correlation_analysis(df)
    elif args.plot:
        plot_correlation_analysis()
    else:
        print("Please specify --generate and/or --plot")
    