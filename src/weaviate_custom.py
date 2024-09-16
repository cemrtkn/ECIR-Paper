import weaviate
from weaviate.classes.query import MetadataQuery

class weaviate_custom:
    def __init__(self):
        pass
    
    def db_connect(self):
        self.client = weaviate.connect_to_custom(
            http_host="weaviatedb.srv.webis.de",  
            http_port=80,                       
            http_secure=False,   
            grpc_host="weaviateinference.srv.webis.de",  
            grpc_port=80,                       
            grpc_secure=False,
            skip_init_checks=True,
        )
        self.collection = self.client.collections.get("Segments")
        
    def db_disconnect(self):
        self.client.close()

    def retrieve(self, query, top_n):
        # get embeddings and put them into a list
        def get_data(objects):
            for o in objects:
                o_with_vector = self.collection.query.fetch_object_by_id(o.uuid, include_vector= True)
                vector = o_with_vector.vector['default']

                top_documents.append(o.properties["segment"])
                top_vectors.append(vector)
                query_similarities.append(o.metadata.certainty)
            return top_documents, top_vectors, query_similarities
        
        top_documents = []
        top_vectors = []
        query_similarities = []

        response = self.collection.query.near_text(
                    query=query,
                    limit=top_n,
                    return_metadata=MetadataQuery(distance=True, certainty=True)
                )
        
        return get_data(response.objects)
    
