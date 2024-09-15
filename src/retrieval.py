import weaviate
from weaviate.classes.query import MetadataQuery

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
print(topic_keys)


response = collection.query.near_text(
                    query="What is RAG?",
                    limit=2,
                    return_metadata=MetadataQuery(distance=True, certainty=True)
                )

for o in response.objects:
    print(o.properties["docid"])
    print(o.properties["title"])
    print(o.properties["segment"])

client.close()