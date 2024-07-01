from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf


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
    '''
    Displays retrieved images along with their distances to the query image.
    '''
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


def load_images(path, return_filenames=False):
  '''
  Loads images with keras.preprocessing.image.load_img for VGG19 pre-processing
  '''
  path = Path(path)
  # Ensures only valid image files are loaded
  img_paths = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png')) \
              + list(path.glob('*.gif'))
  images = []
  filenames = []
  print(f'Loading {len(img_paths)} images')
  for img_path in tqdm(img_paths):
    # load image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    images.append(img)
    filenames.append(img_path.name)
  print('images loaded as' , type(images[0]), 'type')
  
  if return_filenames:
    return images, filenames
  else:
    return images