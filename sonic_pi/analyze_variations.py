import os
from wavEmbed import embed_wav
from codeEmbed import model as code_model
import torch
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def get_acid_variations():
    """Get all acid variations from database2--acid folder"""
    pairs = []
    base_path = "database2_acid"
    
    for variant_dir in os.listdir(base_path):
        if variant_dir.startswith("acid"):
            variant_path = os.path.join(base_path, variant_dir)
            if os.path.isdir(variant_path):
                # Find the code file and wav file
                code_file = next((f for f in os.listdir(variant_path) if f.endswith('.rb')), None)
                wav_file = next((f for f in os.listdir(variant_path) if f.endswith('.wav')), None)
                
                if code_file and wav_file:
                    pairs.append({
                        'name': variant_dir,
                        'code_path': os.path.join(variant_path, code_file),
                        'wav_path': os.path.join(variant_path, wav_file)
                    })
    
    return pairs

def analyze_acid_variations():
    pairs = get_acid_variations()
    code_embeddings = {}
    wav_embeddings = {}
    
    # Compute embeddings
    for pair in pairs:
        with open(pair['code_path'], 'r') as f:
            code_content = f.read()
        code_embeddings[pair['name']] = code_model.encode(code_content, convert_to_tensor=True)
        wav_embeddings[pair['name']] = embed_wav(pair['wav_path']).squeeze()
    
    # Compute pairwise distances
    sample_pairs = list(combinations(pairs, 2))
    results = []
    
    for pair1, pair2 in sample_pairs:
        name1, name2 = pair1['name'], pair2['name']
        
        code_dist = torch.dist(code_embeddings[name1], code_embeddings[name2]).item()
        wav_dist = torch.dist(wav_embeddings[name1], wav_embeddings[name2]).item()
        
        results.append({
            'pair': f"{name1} vs {name2}",
            'code_distance': code_dist,
            'wav_distance': wav_dist
        })
    
    df = pd.DataFrame(results)
    df.to_csv('acid_variations_analysis.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(df['code_distance'], df['wav_distance'])
    plt.xlabel('Distance in Code Space')
    plt.ylabel('Distance in WAV Space')
    plt.title('Code vs WAV Distances for Acid Variations')
    
    # Add annotations for each point
    for i, row in df.iterrows():
        plt.annotate(row['pair'], (row['code_distance'], row['wav_distance']))
    
    plt.savefig('acid_variations_correlation.png')
    plt.close()
    
    return df

if __name__ == "__main__":
    df = analyze_acid_variations()
    print("\nPairwise Distances for Acid Variations:")
    print(df) 