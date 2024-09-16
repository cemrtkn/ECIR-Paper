from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from typing import List
import uvicorn
from weaviate_custom import weaviate_custom
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

weaviate_db = weaviate_custom()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # connect to weaviate db
    weaviate_db.db_connect()
    yield
    # disconnect from the db
    weaviate_db.db_disconnect()


app = FastAPI(lifespan=lifespan)

topics_file_path = './Topics/topics.rag24.test.txt'
topics = {}
with open(topics_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            topics[parts[0]] = parts[1]

topic_values = list(topics.values())

#document_objects = weaviate_db.retrieve(topic_values[0], 1)
#print(document_objects[0])



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    topic = request.query_params.get("topic")
    document_content = ""
    if topic and topic in topic_values:
        query = topic
        try:
            document = weaviate_db.retrieve(query, 1)
            if document:
                document_content = document[0].properties["segment"]  
            else:
                document_content = "Document not found."
        except Exception as e:
            document_content = f"Error retrieving document: {str(e)}"
    
    dropdown_options = "".join([f'<option value="{value}">{value}</option>' for value in topic_values])
    html_content = f"""
    <html>
        <head>
            <title>Topic Selector</title>
        </head>
        <body>
            <h1>Select a Topic</h1>
            <form action="/" method="get">
                <label for="topic">Choose a topic:</label>
                <select id="topic" name="topic">
                    {dropdown_options}
                </select>
                <input type="submit" value="Submit">
            </form>
            <h2>Selected Topic</h2>
            <p>{query}</p>
            <h2>Document</h2>
            <p>{document_content}</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

    

if __name__ == "__main__":
    uvicorn.run("readable_documents_service:app", host="127.0.0.1", port=8001, reload=True)
