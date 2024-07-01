'''
This file assembles the entire modularized image retrieval system that
returns a list of neighbors given a query image.
'''

from pathlib import Path
from model_loader import load_model
from embedder import get_embeddings, load_precomp_embeddings
from knn_retriever import get_knn_for_query

visual_model = load_model()

# Load and embed query image
query_image = [Path('test_image/test_image.png')]
query_embedding = get_embeddings(visual_model, [23], query_image)

# Load candidate embeddings along with their filenames
db_embeddings, db_filenames = load_precomp_embeddings(Path('./dataset/embeddings/23/'), names=True)

knn_images, knn_E, distances = get_knn_for_query(query_embedding, knnbr, train_images, train_E)