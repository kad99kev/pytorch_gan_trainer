import os

def load_requirements():
    with open("requirements.txt", "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    requirements = []
    for line in lines:
        if line:
            requirements.append(line)
    
    return requirements

print(load_requirements())