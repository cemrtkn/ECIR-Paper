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
        response = self.collection.query.near_text(
                    query=query,
                    limit=top_n,
                    return_metadata=MetadataQuery(distance=True, certainty=True)
                )
        return response.objects
