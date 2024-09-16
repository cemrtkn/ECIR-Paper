from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from typing import List
import uvicorn
from weaviate_custom import weaviate_custom
import logging
from utilities import MMR, diversity_ranker, dartboard



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

def rerank(top_documents, top_vectors, query_similarities, top_n = 5):
    reranked_docs_dict = {
        "original": top_documents[:top_n],
        "mmr": [top_documents[i] for i in MMR(top_vectors, query_similarities, 0.3, top_n)],
        "dr": [top_documents[i] for i in diversity_ranker(top_vectors, top_n)],
        "db": [top_documents[i] for i in dartboard(top_vectors, query_similarities, top_n)]
    }
    return reranked_docs_dict


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    topic = request.query_params.get("topic")
    document_content = ""
    if topic and topic in topic_values:
        query = topic
        try:
            # Retrieve multiple documents
            top_documents, top_vectors, query_similarities = weaviate_db.retrieve(query, 20)  # Get more documents for better reranking
            reranked_docs_dict = rerank(top_documents, top_vectors, query_similarities, 5)  # Rerank top 5 documents
            
            if top_documents:
                # Create the HTML structure to display the reranked documents side by side
                table_rows = ""
                for key, docs in reranked_docs_dict.items():
                    # Create a column for each reranked strategy
                    docs_html = "".join([f"<p>{doc}</p>" for doc in docs])
                    table_rows += f"<td><h3>{key.upper()}</h3>{docs_html}</td>"
                
                document_content = f"""
                <table style="width:100%; text-align:left;">
                    <tr>
                        {table_rows}
                    </tr>
                </table>
                """
            else:
                document_content = "No documents found."
        except Exception as e:
            document_content = f"Error retrieving documents: {str(e)}"
    
    dropdown_options = "".join([f'<option value="{value}">{value}</option>' for value in topic_values])
    html_content = f"""
    <html>
        <head>
            <title>Topic Selector</title>
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                td {{
                    width: 25%; /* Make each column take up 25% of the table width */
                    vertical-align: top; /* Align content at the top of each cell */
                    padding: 10px; /* Add padding for better readability */
                    border: 1px solid #ccc; /* Optional: Add borders for better visual separation */
                }}
                h3 {{
                    text-align: center; /* Center the strategy titles */
                }}
            </style>
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
            <h2>Documents</h2>
            <table>
                <tr>
                    {table_rows}
                </tr>
            </table>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)



    

if __name__ == "__main__":
    uvicorn.run("readable_documents_service:app", host="127.0.0.1", port=8001, reload=True)
