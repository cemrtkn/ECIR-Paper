import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr(documents, query, similarity_matrix, lambda_param=0.5, top_n=5):
    # Calculate similarity between documents and the query
    query_similarities = cosine_similarity([query], documents)[0]
    
    # Initialize selected set and candidate set
    selected_indices = []
    candidate_indices = list(range(len(documents)))

    for _ in range(top_n):
        mmr_score = []
        
        for candidate in candidate_indices:
            # Relevance term: similarity to the query
            relevance = query_similarities[candidate]
            
            # Diversity term: similarity to already selected documents
            if selected_indices:
                diversity = max([similarity_matrix[candidate, selected] for selected in selected_indices])
            else:
                diversity = 0  # No selected documents yet, so no diversity penalty
            
            # MMR score: lambda * relevance - (1 - lambda) * diversity
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_score.append((candidate, mmr))
        
        # Select the document with the highest MMR score
        selected = max(mmr_score, key=lambda x: x[1])[0]
        selected_indices.append(selected)
        candidate_indices.remove(selected)
    
    return selected_indices

