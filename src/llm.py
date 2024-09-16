import requests

def llm_request(prompt):
    url = "https://llm.srv.webis.de/api/generate"
    data = {
        "model": "default",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}, {response.text}"
    
def generation(query, data):
    prompt = (
    "You will receive a user query and relevant sections retrieved from a database. "
    "Your task is to generate an answer to the query using only the information provided in the retrieved documents. "
    "Don't be repetitive even if some relevant sections might repeat information. "
    "Make use of all unique information present in the retrieved sections. "
    "Do not include any information beyond what is discussed in the documents. "
    "Don't cite yourself. "
    "Don't mention segment numbers. "
    "If none of the segments answer the question fully say 'No relevant documents for this query'. "
    "User query: {query} "
    )
    prompt = prompt.format(query=query)
    for idx, segment in enumerate(data):
        prompt += "Relevant document" + str(idx+1) + ": " + segment + "\n"
    return llm_request(prompt)