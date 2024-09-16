from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Define the FastAPI app
app = FastAPI()

# Load the model once on startup
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# Define the request body model
class TextRequest(BaseModel):
    text: str

# Endpoint to get the vector for a text
@app.post("/vectorize")
def vectorize_text(request: TextRequest):
    text = request.text
    # Generate the embedding for the input text
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
