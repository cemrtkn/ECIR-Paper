import weaviate_custom
from weaviate.classes.query import MetadataQuery
from utilities import MMR, diversity_ranker, dartboard
from visualization import visualize_rankings_with_tsne, visualize_rankings_with_pca
import json

def export_docs(reranked_docs_dict, base_output_filename):
    mmr_file = f"./output/{base_output_filename}_mmr.jsonl"
    dr_file = f"./output/{base_output_filename}_dr.jsonl"
    db_file = f"./output/{base_output_filename}_db.jsonl"
    
    with open(mmr_file, 'w') as mmr_f, open(dr_file, 'w') as dr_f, open(db_file, 'w') as db_f:
        for topic, rankings in reranked_docs_dict.items():
            mmr_data = {
                "topic": topic,
                "documents": rankings["mmr"]
            }
            dr_data = {
                "topic": topic,
                "documents": rankings["dr"]
            }
            db_data = {
                "topic": topic,
                "documents": rankings["db"]
            }
            
            mmr_f.write(json.dumps(mmr_data) + '\n')
            dr_f.write(json.dumps(dr_data) + '\n')
            db_f.write(json.dumps(db_data) + '\n')


client = weaviate_custom.connect_to_custom(
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

reranked_docs_dict = {}
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
    
    reranked_docs_dict[topic] = { 
    "mmr": [top_documents[i] for i in MMR(top_vectors, query_similarities, 0.3)],
    "dr": [top_documents[i] for i in diversity_ranker(top_vectors)],
    "db": [top_documents[i] for i in dartboard(top_vectors, query_similarities)]
}

    counter += 1
    if counter == 2:
        break

export_docs(reranked_docs_dict, "reranked_docs")

'''print("No reranking \n")

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


visualize_rankings_with_tsne(topic, top_vectors, selected_indices_mmr, selected_indices_dr, selected_indices_db)'''


client.close()