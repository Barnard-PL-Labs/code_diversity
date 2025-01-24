import os
from wavEmbed import embed_wav
from codeEmbed import model as code_model
import torch
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

def get_acid_variations(base_path="dataset_vari", subfolder=None):
    """Get all variations from the specified database folder and subfolder
    
    Args:
        base_path (str): Path to the dataset folder (default: "dataset_vari")
        subfolder (str): Optional subfolder to analyze (e.g., "bach", "acid", etc.)
    """
    pairs = []
    
    if subfolder:
        # Only look in the specified subfolder
        subfolder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(subfolder_path):
            raise ValueError(f"Subfolder {subfolder} not found in {base_path}")
            
        for variant_dir in os.listdir(subfolder_path):
            variant_path = os.path.join(subfolder_path, variant_dir)
            if os.path.isdir(variant_path):
                code_file = next((f for f in os.listdir(variant_path) if f.endswith('.rb') or f.endswith('.pi')), None)
                wav_file = next((f for f in os.listdir(variant_path) if f.endswith('.wav')), None)
                
                if code_file and wav_file:
                    pairs.append({
                        'name': variant_dir,
                        'code_path': os.path.join(variant_path, code_file),
                        'wav_path': os.path.join(variant_path, wav_file)
                    })
    return pairs

def analyze_variations(database_path="dataset_vari", subfolder=None):
    """Analyze variations in the dataset
    
    Args:
        database_path (str): Path to the dataset folder (default: "dataset_vari")
        subfolder (str): Optional subfolder to analyze (e.g., "bach", "acid", etc.)
    """
    # Import models only when analyzing
    from wavEmbed import compute_wav_similarity
    from codeEmbed import compute_code_similarity
    
    pairs = get_acid_variations(database_path, subfolder)
    if not pairs:
        print(f"No valid pairs found in {subfolder if subfolder else database_path}")
        return None
        
    # Store raw code content and embeddings
    code_contents = {}
    wav_embeddings = {}
    
    # Read code contents first
    for pair in pairs:
        with open(pair['code_path'], 'r') as f:
            code_contents[pair['name']] = f.read()
        wav_embeddings[pair['name']] = embed_wav(pair['wav_path']).squeeze()
    
    # Compute pairwise distances
    sample_pairs = list(combinations(pairs, 2))
    results = []
    
    for pair1, pair2 in sample_pairs:
        name1, name2 = pair1['name'], pair2['name']
        
        # Pass raw code content instead of embeddings
        code_sim = compute_code_similarity(code_contents[name1], code_contents[name2])
        wav_sim = compute_wav_similarity(pair1['wav_path'], pair2['wav_path'])
        
        results.append({
            'pair': f"{name1} vs {name2}",
            'code_distance': code_sim,
            'wav_distance': wav_sim,
            'subfolder': subfolder
        })
    
    return pd.DataFrame(results)

def generate_variation_data(base_path="dataset_vari"):
    """Generate and save the variation analysis data"""
    # Get all subfolders
    subfolders = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    # Collect results from all subfolders
    all_results = []
    for subfolder in subfolders:
        print(f"Analyzing {subfolder}...")
        df = analyze_variations(subfolder=subfolder)
        if df is not None:
            all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv('variations_analysis_all.csv', index=False)
    return combined_df
def plot_variation_analysis(combined_df=None):
    """Generate plots from variation analysis data"""
    if combined_df is None:
        # Load data if not provided
        combined_df = pd.read_csv('variations_analysis_all.csv')
    
    # Calculate correlations
    pearson_corr, p_value_pearson = pearsonr(combined_df['code_distance'], combined_df['wav_distance'])
    spearman_corr, p_value_spearman = spearmanr(combined_df['code_distance'], combined_df['wav_distance'])
    
    # Set up Computer Modern font with bold weight
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.weight'] = 'bold'
    
    # Create single figure with larger size
    plt.figure(figsize=(12, 12))
    
    # Plot points with different colors for each pair
    unique_pairs = combined_df['pair'].unique()
    # Get pairs with 'original' for distinct colors
    original_pairs = [pair for pair in unique_pairs if 'original' in pair]
    # Generate distinct colors with darker shades
    original_colors = ['red', 'green', 'blue']
    
    # Create color mapping: dark blue for non-original, distinct colors for original
    color_map = {}
    original_idx = 0
    for pair in unique_pairs:
        if 'original' in pair:
            color_map[pair] = original_colors[original_idx]
            original_idx += 1
        else:
            color_map[pair] = 'gray' 
            
    # Determine the global min and max for both axes
    min_val = min(combined_df['code_distance'].min(), combined_df['wav_distance'].min())
    max_val = max(combined_df['code_distance'].max(), combined_df['wav_distance'].max())
    
    # Set axis limits before plotting points
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Create mapping for custom labels
    custom_labels = {
        'v1 vs original': 'sleep',
        'v2 vs original': 'amp',
        'v3 vs original': 'bpm',
        'v1 vs v2': 'sleep vs amp',
        'v1 vs v3': 'sleep vs bpm',
        'v3 vs v2': 'bpm vs amp',
    }
    
    base_fontsize = 40

    # Plot points with higher z-order
    # Sort pairs by length of their custom labels (or original pair if no custom label)
    sorted_pairs = sorted(unique_pairs, 
                         key=lambda x: len(custom_labels.get(x, x)))
    
    for pair in sorted_pairs:  # Changed from unique_pairs to sorted_pairs
        subset = combined_df[combined_df['pair'] == pair]
        plt.scatter(subset['code_distance'], subset['wav_distance'], 
                   alpha=0.6, label=custom_labels.get(pair, pair),
                   color=color_map[pair],
                   s=230, zorder=2)
    
    plt.legend(loc='lower left', fontsize=base_fontsize)
    
    # Add labels and title with increased font sizes
    plt.xlabel('Cosine Similarity in Code Space', fontsize=base_fontsize, fontweight='bold')
    plt.ylabel('Cosine Similarity in WAV Space', fontsize=base_fontsize, fontweight='bold')
    plt.title('Code and WAV Similarities on Variations', fontsize=base_fontsize, fontweight='bold')

    # Increase tick label sizes
    plt.xticks(fontsize=base_fontsize, fontweight='bold')
    plt.yticks(fontsize=base_fontsize, fontweight='bold')

    plt.tight_layout()
    plt.savefig('variations_correlation_all.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOverall Correlation Analysis:")
    print(f"Pearson correlation: {pearson_corr:.3f} (p={p_value_pearson:.4f})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p={p_value_spearman:.4f})")
    
    print("\nSummary by subfolder:")
    for subfolder in combined_df['subfolder'].unique():
        subset = combined_df[combined_df['subfolder'] == subfolder]
        corr, p_val = pearsonr(subset['code_distance'], subset['wav_distance'])
        print(f"{subfolder}: Pearson r={corr:.3f} (p={p_val:.4f}, n={len(subset)})")

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate new analysis data')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    if args.generate:
        combined_df = generate_variation_data()
        if args.plot:
            plot_variation_analysis(combined_df)
    elif args.plot:
        plot_variation_analysis()
    else:
        print("Please specify --generate and/or --plot") 