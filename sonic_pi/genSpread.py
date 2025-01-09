import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
from itertools import combinations
from codeEmbed import compute_code_similarity

load_dotenv()

def generate_solution(prompt: str) -> str:
    """Generate a single Python solution for a given coding problem
    
    Args:
        prompt (str): Natural language description of the coding problem
        
    Returns:
        str: The Python code solution
    """
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Initialize the client
    client = OpenAI()
    
    # Updated system prompt for Python coding problems
    system_prompt = """You are an expert Python programmer. Given a coding problem description, 
    generate a Python solution that:
    1. Uses idiomatic Python syntax and best practices
    2. Includes clear comments explaining the approach
    3. Is complete and runnable
    4. Considers time and space complexity
    
    Return ONLY the code, no explanations or additional text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Encourage creativity while maintaining coherence
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating solution: {str(e)}")
        return ""

def generate_solutions(prompt: str, n: int = 3) -> List[str]:
    """Generate n different Python solutions and compute their similarities
    
    Args:
        prompt (str): Natural language description of the coding problem
        n (int): Number of solutions to generate (default: 3)
        
    Returns:
        List[str]: List of Python code solutions
    """
    solutions = []
    for i in range(n):
        solution = generate_solution(prompt)
        if solution:
            solutions.append(solution)
    
    # Compute similarities between all pairs of solutions
    if len(solutions) > 1:
        print("\nCode Similarities:")
        for (i, sol1), (j, sol2) in combinations(enumerate(solutions, 1), 2):
            similarity = compute_code_similarity(sol1, sol2)
            print(f"Solution {i} vs Solution {j}: {similarity:.4f}")
    
    return solutions

def explain_code(code: str) -> str:
    """Generate a natural language explanation of what the code does
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        str: Natural language explanation of the code
    """
    
    client = OpenAI()
    
    system_prompt = """You are an expert Python programmer. Analyze the given code and provide one line explanation what the code is doing. Do not include any interpretation of algorithms or approaches.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain this code:\n\n{code}"}
            ],
            temperature=0.3,  # Lower temperature for more focused explanation
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return ""

if __name__ == "__main__":
    # Example using different approaches to find the nth Fibonacci number
    prompt = """Write a function to find the nth Fibonacci number (where fib(0) = 0, fib(1) = 1). 
    Be creative in your approach - don't use the same method as other solutions."""
    
    solutions = generate_solutions(prompt, 5)
    explanations = [explain_code(sol) for sol in solutions]
    
    # Compare code similarities
    print("\nCode Similarities:")
    for (i, sol1), (j, sol2) in combinations(enumerate(solutions, 1), 2):
        similarity = compute_code_similarity(sol1, sol2)
        print(f"Code {i} vs Code {j}: {similarity:.4f}")
    
    # Compare explanation similarities
    print("\nExplanation Similarities:")
    for (i, exp1), (j, exp2) in combinations(enumerate(explanations, 1), 2):
        similarity = compute_code_similarity(exp1, exp2)
        print(f"Explanation {i} vs Explanation {j}: {similarity:.4f}")
    
    # Print all solutions with their explanations
    for i, (solution, explanation) in enumerate(zip(solutions, explanations), 1):
        print(f"\nSolution {i}:")
        print(solution)
        print("\nExplanation:")
        print(explanation)
        print("\n" + "="*50)