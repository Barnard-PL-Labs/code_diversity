import os
from pathlib import Path
from codeEmbed import compute_code_similarity

def get_acid_files():
    """Get all Ruby files from acid subfolders."""
    base_path = Path('dataset_vari/acid')
    code_files = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.rb') or file.endswith('.pi'):
                code_files.append(os.path.join(root, file))
    
    return code_files

def main():
    # Get all code files
    code_files = get_acid_files()
    
    # Read file contents
    file_contents = {}
    for file_path in code_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_contents[file_path] = f.read()
    
    # Compare each file with every other file
    print("\nCode Similarity Analysis:")
    print("-" * 50)
    
    for i, file1 in enumerate(code_files):
        for file2 in code_files[i+1:]:
            similarity = compute_code_similarity(
                file_contents[file1],
                file_contents[file2]
            )
            print(f"{os.path.basename(file1)} <-> {os.path.basename(file2)}: {similarity:.4f}")

if __name__ == "__main__":
    main()
