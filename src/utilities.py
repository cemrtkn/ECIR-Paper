from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def similarity_matrix_calculator(vectors):

    X = np.array(vectors)
    similarity_matrix = cosine_similarity(X)

    return similarity_matrix

import numpy as np

# Function to calculate the natural log of Gaussian PDF
def LogNorm(arr, sigma):
    """
    Compute the natural log of a Gaussian probability density function.
    
    Parameters:
    arr : numpy array
        Input array representing the data points (D or Q).
    sigma : float
        The standard deviation (sigma) of the Gaussian distribution.
        
    Returns:
    numpy array
        The transformed array after applying the log of Gaussian PDF.
    """
    ln_2pi = np.log(2 * np.pi)
    term1 = np.log(sigma)
    term2 = -0.5 * ln_2pi
    term3 = -0.5 * (arr ** 2) / (sigma ** 2)
    
    return term1 + term2 + term3


def MMR(vectors, query_similarities ,lambda_param=0.5, top_n=5):
    # smaller the lambda value higher diversity there is
    similarity_matrix = similarity_matrix_calculator(vectors)
    
    selected_indices = []
    candidate_indices = list(range(len(vectors)))

    for _ in range(top_n):
        mmr_score = []
        
        for candidate in candidate_indices:
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



def diversity_ranker(doc_vectors, top_n=5):
    
    similarity_matrix = similarity_matrix_calculator(doc_vectors)
    
    # the vectors are already relevance sorted
    first_selected_idx = 0
    ranked_docs = [first_selected_idx]
    
    # Track remaining documents that haven't been selected
    remaining_indices = set(range(len(doc_vectors))) - {first_selected_idx}
    
    # Step 3: Iteratively select documents based on average similarity to selected documents
    while remaining_indices:
        avg_similarities = []
        
        # Calculate the average similarity of each remaining document to the already selected documents
        for idx in remaining_indices:
            # Get the similarities of the current document to all selected documents
            selected_sims = similarity_matrix[idx, ranked_docs]
            
            # Compute average similarity to selected documents
            avg_similarity = np.mean(selected_sims)
            avg_similarities.append((idx, avg_similarity))
        
        # Step 4: Select the document with the lowest average similarity
        next_selected_idx = min(avg_similarities, key=lambda x: x[1])[0]
        ranked_docs.append(next_selected_idx)
        
        # Update remaining documents
        remaining_indices.remove(next_selected_idx)
    
    return ranked_docs[:top_n]

# Function to perform the Dartboard algorithm
def dartboard(doc_vectors, query_similarities, top_n = 5, sigma =  0.096):
    """
    Greedily seed and search for the most relevant k points based on similarity.
    
    Parameters:
    D : numpy array
        nxn matrix for similarity between documents.
    Q : numpy array
        A list of cosine similarities between the query and all the documents.
    k : int
        Number of documents to return.
    sigma : float
        The standard deviation of the Gaussians (a measure of spread).
        
    Returns:
    list
        List of indices of the top k selected documents.
    """
    query_similarities = np.array(query_similarities)
    D = similarity_matrix_calculator(doc_vectors)
    # Step 1: Apply LogNorm transformation to D and Q
    D_ln = LogNorm(D, sigma)
    Q_ln = LogNorm(query_similarities, sigma)
    
    # Step 2: Greedily seed and search
    # the docs are relevance ranked
    m = 0
    maxes = D_ln[m]  
    ret = [m]  # Initialize return list with index m
    
    # Step 3: Incrementally add until we have k elements
    while len(ret) < top_n:
        # Find the new maximum for the selected document
        newmax = np.maximum(maxes, D_ln)  # Element-wise maximum of maxes and D
        scores = np.log(np.sum(np.exp(newmax + Q_ln), axis=1))  # LogSumExp calculation
        
        m = np.argmax(scores)  # Get index of max score
        maxes = newmax[m]  # Update maxes with the row corresponding to m
        ret.append(m)  # Append the new index to the return list
    
    return ret
