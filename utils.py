from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def index_files(dir, prefix='wb'):
    path = Path(dir)
    files = list(path.glob('*.*'))  # Get all files regardless of extension
    for i, file in enumerate(files):
        new_filename = f'{prefix}_{i}.jpg'
        if file != path / new_filename:
            file.rename(path / new_filename)
    print(f'Renamed {len(files)} files in {dir}')

