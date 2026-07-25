import subprocess
import os
import json

pdfs_dir = 'Project1_ref_papers'
output = {}

for f in os.listdir(pdfs_dir):
    if f.endswith('.pdf'):
        path = os.path.join(pdfs_dir, f)
        result = subprocess.run(['mdimport', '-t', '-d3', path], capture_output=True, text=True)
        # The output contains kMDItemTextContent = "..."
        lines = result.stdout.split('\n')
        text_content = ""
        in_text = False
        for line in lines:
            if "kMDItemTextContent =" in line:
                in_text = True
                text_content += line.replace('kMDItemTextContent = "', '') + "\n"
            elif in_text:
                if line.strip() == '";' or line.strip() == 'kMDItemVersion = "1.7";':
                    in_text = False
                    break
                text_content += line + "\n"
        output[f] = text_content[:15000] # Take first 15000 chars

with open('/Users/shubham_kumar/.gemini/antigravity-ide/scratch/all_papers.json', 'w') as f:
    json.dump(output, f)
print("Extracted PDF texts.")
