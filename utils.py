from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def index_files(dir, prefix='wb'):
    '''
    Indexes files in a directory by renaming them with a prefix and a number.
    '''
    path = Path(dir)
    files = list(path.glob('*.*'))  # Get all files regardless of extension
    for i, file in enumerate(files):
        new_filename = f'{prefix}_{i}.jpg'
        if file != path / new_filename:
            file.rename(path / new_filename)
    print(f'Renamed {len(files)} files in {dir}')


def show_images(images, distances, pair_distances_img, pair_distances_emb, norm=False):
    # Display image in new window
    fig = plt.figure(figsize=(14, 8))

    # Normalizes all distances if norm is True
    if norm:
      distances = distances / np.max(distances)
      pair_distances_img = pair_distances_img / np.max(pair_distances_img)
      pair_distances_emb = pair_distances_emb / np.max(pair_distances_emb)

    for i in range(len(images)):
      ax = fig.add_subplot(1, len(images), i+1)
      if i == 0:
        ax.set_title('Query Image')
        # show distance
        ax.text(0.5, -0.1, f'VGG distance to query', ha='center', va='center', transform=ax.transAxes)
        # show pair distance img
        ax.text(0.5, -0.2, f'SSIM to image on the left', ha='center', va='center', transform=ax.transAxes)
        # show pair distance emb
        ax.text(0.5, -0.3, f'VGG distance to image on the left', ha='center', va='center', transform=ax.transAxes)
      if i >= 1:
        title = 'neighbor' + str(i)
        ax.set_title(title)
        # show distance
        ax.text(0.5, -0.1, f'{distances[i-1]:.2f}', ha='center', va='center', transform=ax.transAxes)
      if i >= 2:
        # show pair distance img
        ax.text(0.5, -0.2, f'{pair_distances_img[i-2]:.2f}', ha='center', va='center', transform=ax.transAxes)
        # show pair distance emb
        ax.text(0.5, -0.3, f'{pair_distances_emb[i-2]:.2f}', ha='center', va='center', transform=ax.transAxes)

      plt.imshow(np.squeeze(images[i]))
      plt.axis('off')
    plt.show()