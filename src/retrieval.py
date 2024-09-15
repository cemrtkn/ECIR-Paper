import weaviate
from weaviate.classes.query import MetadataQuery
from utilities import MMR, diversity_ranker, dartboard
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


client = weaviate.connect_to_custom(
    http_host="weaviatedb.srv.webis.de",  
    http_port=80,                       
    http_secure=False,   
    grpc_host="weaviateinference.srv.webis.de",  
    grpc_port=80,                       
    grpc_secure=False,
    skip_init_checks=True,
)

collection=client.collections.get("Segments")

topics_file_path = './Topics/topics.rag24.test.txt'

topics = {}

with open(topics_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            topics[parts[0]] = parts[1]

topic_keys = list(topics.keys())


counter = 0
for topic_id in topic_keys:
    top_documents = []
    top_vectors = []
    query_similarities = []

    topic = topics[topic_id]
    print(topic , '\n')

    response = collection.query.near_text(
                    query=topic,
                    limit=20,
                    return_metadata=MetadataQuery(distance=True, certainty=True)
                )
    for o in response.objects:
        o_with_vector = collection.query.fetch_object_by_id(o.uuid, include_vector= True)
        vector = o_with_vector.vector['default']

        top_documents.append(o.properties["segment"])
        top_vectors.append(vector)
        query_similarities.append(o.metadata.certainty)

    counter += 1

    if counter == 1:
        break


selected_indices_mmr = MMR(top_vectors, query_similarities, 0.3)
selected_indices_dr = diversity_ranker(top_vectors)
selected_indices_db = dartboard(top_vectors, query_similarities)

print("No reranking \n")

for id in range(5):
    #print(top_documents[id], '\n')
    print(query_similarities[id])

print()

print("Selected by MMR: ", selected_indices_mmr, '\n')

for id in selected_indices_mmr:
    #print(top_documents[id], '\n')
    print(query_similarities[id])

print("Selected by DiversityRanker: ", selected_indices_dr, '\n')
for id in selected_indices_dr:
    #print(top_documents[id], '\n')
    print(query_similarities[id])

print("Selected by Dartboard: ", selected_indices_db, '\n')
for id in selected_indices_db:
    #print(top_documents[id], '\n')
    print(query_similarities[id])


# Create sets for easier comparison
mmr_set = set(selected_indices_mmr)
dr_set = set(selected_indices_dr)
db_set = set(selected_indices_db)

# Initialize an array to store the color category for each document
colors = np.zeros(len(top_vectors))

# Assign color codes based on the sets
for i in range(len(top_vectors)):
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

# Apply t-SNE
tsne_vectors = np.array(top_vectors)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(tsne_vectors)

# Create a color map
colormap = plt.cm.get_cmap('tab10', 8)  # Using a colormap with 8 distinct colors

# Plot the results with color coding
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap=colormap, s=100)

# Create a legend for the color categories
legend_labels = ['Not selected', 'DB only', 'DR only', 'MMR only', 'DR & DB', 'MMR & DB', 'MMR & DR', 'All three']
legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Selection")

plt.title('t-SNE Visualization of Document Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

client.close()