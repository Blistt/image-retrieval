'''
This file implements a function to retrieve the k nearest neighbors of a query image
using a pre-fit KNN model from the sklearn library.
'''

def get_knn_for_query(e, knnbr, train_images, train_embeddings):
  e = e.reshape(1, -1)

  # Get the neighbors of the query image
  knn = knnbr.kneighbors(e)
            
  # GETS THE IMAGES OF THE KNN
  knn_images = []
  # Retrieve the k nearest neighbor images of the query image 
  for i in knn[1][0][1:]:
    knn_images.append(train_images[i])

  # GETS THE EMBEDDINGS OF THE KNN
  knn_E = []
  # Retrieve the k nearest neighbor embeddings of the query image 
  for i in knn[1][0][1:]:
    knn_E.append(train_embeddings[i])


  return knn_images, knn_E