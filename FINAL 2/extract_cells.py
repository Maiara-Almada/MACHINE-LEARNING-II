import json
import sys

try:
    with open('final_notebook.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    
    print("--- LAST 10 CODE CELLS ---")
    for i, cell in enumerate(code_cells[-10:]):
        print(f"\n--- CELL {i+1} ---")
        print(''.join(cell))
except Exception as e:
    print(f"Error: {e}")
