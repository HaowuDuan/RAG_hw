from config import EMBEDDING_URL
import requests
import time
import traceback

def local_embedding(inputs):
    """Get embeddings from the embedding service"""
    
    headers = {"Content-Type": "application/json"}
    data = {"texts": inputs}
    
    response = requests.post(EMBEDDING_URL, headers=headers, json=data)
    
    result = response.json()
    return result['data']['text_vectors']

def openai_embedding(inputs):
    pass

if __name__ == '__main__':
    inputs = ["Hello, world!"]
    output = local_embedding(inputs)[0]
    print(output)
    print("Dim: ",len(output))
    