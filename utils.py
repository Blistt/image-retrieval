from tqdm import tqdm
import os
import numpy as np
import tensorflow as tf

def load_images(images, path, view_images=[]):
  '''
  Load images from the given path and preprocess them for VGG19 model
  '''
  img_paths = os.listdir(path)
  print(f'Loading {len(img_paths)} images')
  for img in tqdm(img_paths):
    img = tf.keras.preprocessing.image.load_img(os.path.join(path,img), target_size=(224,224))
    view_images.append(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)       # Pre-processes image for VGG19
    images.append(img)
  return len(os.listdir(path))

import matplotlib.pyplot as plt

def show_images(images, k=6):
    '''
    Loads a query image
    '''
    # Display image in new window
    fig = plt.figure(figsize=(14, 8))

    for i in range(1, len(images)+1):
      ax = fig.add_subplot(len(images)//6, 6, i)
      if i == 1:
        ax.set_title('Query Image')
      else:
        title = 'neighbor' + str(i-1)
        ax.set_title(title)

      plt.imshow(images[i-1])
      plt.axis('off')
    plt.show()