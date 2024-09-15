import weaviate
from weaviate.classes.query import MetadataQuery
from utilities import mmr


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

    if counter == 3:
        break


selected_indices = mmr(top_vectors, query_similarities)
print(selected_indices, '\n')

for id in selected_indices:
    print(top_documents[id], '\n')

client.close()