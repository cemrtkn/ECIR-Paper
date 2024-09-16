import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests

embedding_url = "http://127.0.0.1:8000/vectorize"

def visualize_rankings_with_tsne(query, top_vectors, selected_indices_mmr, selected_indices_dr, selected_indices_db, perplexity=5):
    """
    Visualizes document embeddings using t-SNE and colors them based on their selection in different ranking methods.

    Args:
    - query (str): The query text to be vectorized and visualized.
    - top_vectors (array-like): The 384-dimensional embeddings for the documents.
    - selected_indices_mmr (list): Indices of documents selected by MMR.
    - selected_indices_dr (list): Indices of documents selected by the Diversity Ranker.
    - selected_indices_db (list): Indices of documents selected by Dartboard.
    - perplexity (int): The perplexity parameter for t-SNE (default is 5).

    Returns:
    - None: Displays the 2D t-SNE plot with color coding based on ranking selection and query embedding in red.
    """
    # Convert the selected indices to sets for easier comparison
    mmr_set = set(selected_indices_mmr)
    dr_set = set(selected_indices_dr)
    db_set = set(selected_indices_db)

    # Initialize an array to store the color category for each document
    num_vectors = len(top_vectors)
    colors = np.zeros(num_vectors)

    # Assign color codes based on the sets
    for i in range(num_vectors):
        in_mmr = i in mmr_set
        in_dr = i in dr_set
        in_db = i in db_set

        # Assign a unique number for each combination
        if in_mmr and in_dr and in_db:
            colors[i] = 7  # Selected by all three
        elif in_mmr and in_dr:
            colors[i] = 6  # Selected by MMR and DR
        elif in_mmr and in_db:
            colors[i] = 5  # Selected by MMR and DB
        elif in_dr and in_db:
            colors[i] = 4  # Selected by DR and DB
        elif in_mmr:
            colors[i] = 3  # Selected only by MMR
        elif in_dr:
            colors[i] = 2  # Selected only by DR
        elif in_db:
            colors[i] = 1  # Selected only by DB
        else:
            colors[i] = 0  # Not selected by any

    # Get the query embedding
    query_data = {"text": query}
    response = requests.post(embedding_url, json=query_data)
    query_embedding = response.json()["embedding"]

    # Append the query_embedding to top_vectors for t-SNE
    all_vectors = np.vstack([top_vectors, query_embedding])

    # Apply t-SNE to both document embeddings and the query embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(all_vectors))

    # Extract the 2D position of the query embedding (last one)
    query_2d = embeddings_2d[-1]

    # Create a color map
    colormap = plt.cm.get_cmap('tab10', 8)  # Using a colormap with 8 distinct colors

    # Plot the document embeddings with color coding
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], c=colors, cmap=colormap, s=100)

    # Plot the query embedding separately in red
    plt.scatter(query_2d[0], query_2d[1], color='red', marker='x', s=200, label="Query")

    # Add text annotations for each point (display the index number)
    for i in range(len(embeddings_2d) - 1):
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i), fontsize=9, ha='right')

    # Create a legend for the color categories
    legend_labels = ['Not selected', 'DB only', 'DR only', 'MMR only', 'DR & DB', 'MMR & DB', 'MMR & DR', 'All three']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Selection")
    plt.legend(loc="best")

    plt.title('t-SNE Visualization of Document Embeddings with Query')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


def visualize_rankings_with_pca(query, top_vectors, selected_indices_mmr, selected_indices_dr, selected_indices_db):
    """
    Visualizes document embeddings using PCA and colors them based on their selection in different ranking methods.

    Args:
    - query (str): The query text to be vectorized and visualized.
    - top_vectors (array-like): The 384-dimensional embeddings for the documents.
    - selected_indices_mmr (list): Indices of documents selected by MMR.
    - selected_indices_dr (list): Indices of documents selected by the Diversity Ranker.
    - selected_indices_db (list): Indices of documents selected by Dartboard.

    Returns:
    - None: Displays the 2D PCA plot with color coding based on ranking selection and query embedding in red.
    """
    # Convert the selected indices to sets for easier comparison
    mmr_set = set(selected_indices_mmr)
    dr_set = set(selected_indices_dr)
    db_set = set(selected_indices_db)

    # Initialize an array to store the color category for each document
    num_vectors = len(top_vectors)
    colors = np.zeros(num_vectors)

    # Assign color codes based on the sets
    for i in range(num_vectors):
        in_mmr = i in mmr_set
        in_dr = i in dr_set
        in_db = i in db_set

        # Assign a unique number for each combination
        if in_mmr and in_dr and in_db:
            colors[i] = 7  # Selected by all three
        elif in_mmr and in_dr:
            colors[i] = 6  # Selected by MMR and DR
        elif in_mmr and in_db:
            colors[i] = 5  # Selected by MMR and DB
        elif in_dr and in_db:
            colors[i] = 4  # Selected by DR and DB
        elif in_mmr:
            colors[i] = 3  # Selected only by MMR
        elif in_dr:
            colors[i] = 2  # Selected only by DR
        elif in_db:
            colors[i] = 1  # Selected only by DB
        else:
            colors[i] = 0  # Not selected by any

    # Get the query embedding
    query_data = {"text": query}
    response = requests.post(embedding_url, json=query_data)
    query_embedding = response.json()["embedding"]

    # Append the query_embedding to top_vectors for PCA
    all_vectors = np.vstack([top_vectors, query_embedding])

    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(np.array(all_vectors))

    # Extract the 2D position of the query embedding (last one)
    query_2d = embeddings_2d[-1]

    # Create a color map
    colormap = plt.cm.get_cmap('tab10', 8)  # Using a colormap with 8 distinct colors

    # Plot the document embeddings with color coding
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], c=colors, cmap=colormap, s=100)

    # Plot the query embedding separately in red
    plt.scatter(query_2d[0], query_2d[1], color='red', marker='x', s=200, label="Query")

    # Add text annotations for each point (display the index number)
    for i in range(len(embeddings_2d) - 1):
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i), fontsize=9, ha='right')

    # Create a legend for the color categories
    legend_labels = ['Not selected', 'DB only', 'DR only', 'MMR only', 'DR & DB', 'MMR & DB', 'MMR & DR', 'All three']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Selection")
    plt.legend(loc="best")

    plt.title('PCA Visualization of Document Embeddings with Query')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.show()
