import os
import json
import base64
import mimetypes
import time
import traceback
import logging
import requests
from config import get_es, IMAGE_MODEL_URL, EMBEDDING_URL, RERANK_URL
from embedding import local_embedding
from openai import OpenAI

def _truncate(text: str, max_len: int = 1500) -> str:
    """Truncate text to specified length"""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"

def summarize_image_with_local_model(image_path, base_url=IMAGE_MODEL_URL):
    """Generate image description using local vision model"""
    retry = 0
    while retry <= 5:
        try:
            prompt = """
Describe the content of this image in detail, don't miss any details, and extract any text from the image. Please only objectively describe the image content without any evaluation.
"""
            
            client = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)

            # Read local image and convert to Base64 data URL
            with open(image_path, 'rb') as f:
                content_bytes = f.read()
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/png'
            encoded = base64.b64encode(content_bytes).decode('utf-8')
            data_url = f"data:{mime_type};base64,{encoded}"
            
            resp = client.chat.completions.create(
                model='internvl-internlm2',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt}, 
                        {'type': 'image_url', 'image_url': {'url': data_url}}
                    ]
                }], 
                temperature=0.8, 
                top_p=0.8, 
                max_tokens=2048, 
                stream=False
            )
            
            return resp.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            logging.error(traceback.format_exc())
            time.sleep(1)
            retry += 1
            
    return None

def context_augmentation(page_context, image_description):
    """Augment image description using page context"""
    try:
        # Use local model for context augmentation
        client = OpenAI(api_key='YOUR_API_KEY', base_url=IMAGE_MODEL_URL)
        
        prompt = f'''
Objective: Enhance the image description details through the image's context and source file information, accurately describing the actual content and purpose of the image in the document.

Important Notes:
- There may be noise in the context, please filter carefully.
- Focus on image captions in the context as they usually describe the purpose and meaning of the image.
- Preserve the image's intent and important information, filter out information unrelated to the context.
- Sometimes repetitive content may appear in image descriptions, treat such content as noise and filter it out.
- Please output the answer directly without explanation.
- If the image contains no content or is a background image, output 0

Expected Output:
- A precise and detailed description explaining the role and significance of the image in the context, as well as important details in the image.
- Preserve text and data from the image description, do not omit anything.

Image Description:
```
{image_description}
```

Context:
```
{page_context}
```
'''
        
        response = client.chat.completions.create(
            model="internvl-internlm2",
            messages=[
                {"role": "system", "content": "You are an intelligent AI assistant that enhances image descriptions based on context. The enhanced descriptions should be more accurate, detailed, and complete."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error in context augmentation: {e}")
        return image_description  # Return original description if augmentation fails

def process_existing_images(images_dir="images", es_index="hw_index"):
    """Process existing images from the images directory and index to Elasticsearch"""
    es = get_es()
    
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} not found!")
        return []
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process...")
    
    results = []
    
    for image_file in image_files:
        try:
            image_path = os.path.join(images_dir, image_file)
            print(f"Processing {image_file}...")
            
            # Extract page number from filename (assuming format: filename_pageX_imgY.png)
            page_num = 0
            if '_page' in image_file:
                try:
                    page_part = image_file.split('_page')[1].split('_')[0]
                    page_num = int(page_part) - 1  # Convert to 0-based
                except:
                    pass
            
            # Generate initial description using local vision model
            description = summarize_image_with_local_model(image_path)
            
            if description is None:
                print(f"Failed to generate description for {image_file}")
                continue
            
            # For context augmentation, we would need the page text
            # Since we don't have the original PDF context here, we'll use the description as-is
            # Or you could modify this to extract context from a separate source
            enhanced_description = description  # Could add context augmentation here
            
            # Create embedding using local embedding service
            embedding = local_embedding([enhanced_description])
            if embedding is None or len(embedding) == 0:
                print(f"Failed to get embedding for {image_file}")
                continue
            
            # Prepare document for Elasticsearch
            doc_body = {
                'text': enhanced_description,
                'vector': embedding[0],
                'type': 'image',
                'page': page_num + 1,
                'image_id': image_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', ''),
                'image_path': image_path,
                'source_pdf': 'LLM_fintuing_guide.pdf'  # Update this as needed
            }
            
            # Index to Elasticsearch
            retry = 0
            while retry <= 5:
                try:
                    es.index(index=es_index, body=doc_body)
                    print(f"✓ Indexed {image_file}")
                    break
                except Exception as e:
                    print(f'[Elastic Error] {str(e)} retry {retry}')
                    retry += 1
                    time.sleep(1)
            
            results.append({
                "image_file": image_file,
                "image_path": image_path,
                "page_num": page_num + 1,
                "description": enhanced_description,
                "indexed": True
            })
            
        except Exception as e:
            logging.error(f"Error processing {image_file}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    print(f"Processed {len(results)} images successfully")
    return results

def search_images(query, es_index="hw_index", top_k=5):
    """Search for images using text query"""
    es = get_es()
    
    # Get embedding for the query
    query_embedding = local_embedding([query])
    if query_embedding is None or len(query_embedding) == 0:
        print("Failed to get query embedding")
        return []
    
    # Vector search for images only
    vector_query = {
        "bool": {
            "must": [
                {"term": {"type": "image"}}  # Only search images
            ],
            "should": [
                {"script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_embedding[0]}
                    }
                }}
            ]
        }
    }
    
    try:
        response = es.search(
            index=es_index,
            query=vector_query,
            size=top_k,
            _source=["text", "image_path", "page", "image_id", "source_pdf"]
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'text': hit['_source'].get('text', ''),
                'image_path': hit['_source'].get('image_path', ''),
                'page': hit['_source'].get('page', 0),
                'image_id': hit['_source'].get('image_id', ''),
                'source_pdf': hit['_source'].get('source_pdf', '')
            })
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def rerank_images(query, image_results):
    """Rerank image search results using the rerank service"""
    try:
        if not image_results:
            return image_results
            
        # Prepare documents for reranking (just the text descriptions)
        documents = [doc['text'] for doc in image_results]
        
        # Call rerank service
        res = requests.post(RERANK_URL, json={"query": query, "documents": documents}).json()
        
        if res and 'scores' in res and len(res['scores']) == len(image_results):
            # Add rerank scores to results
            for idx, doc in enumerate(image_results):
                image_results[idx]['rerank_score'] = res['scores'][idx]
            
            # Sort by rerank score (highest first)
            image_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
        return image_results
        
    except Exception as e:
        print(f"Rerank error: {e}")
        return image_results  # Return original results if reranking fails

def interactive_image_search():
    """Interactive image search interface"""
    print("=== Image Search Interface ===")
    print("Enter your search query (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nImage search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                print("Please enter a search query.")
                continue
            
            print(f"\nSearching images for: '{query}'")
            print("=" * 60)
            
            # Perform image search
            search_results = search_images(query, top_k=10)  # Get more results for reranking
            
            if not search_results:
                print("No images found.")
                continue
            
            print(f"Found {len(search_results)} images:")
            print("=== Initial Vector Search Results ===")
            for i, result in enumerate(search_results[:5], 1):  # Show top 5 initial results
                print(f"\n{i}. Image: {result['image_id']}")
                print(f"   Vector Score: {result['score']:.3f}")
                print(f"   Page: {result['page']}")
                print(f"   Description: {result['text'][:200]}...")
                print(f"   Path: {result['image_path']}")
                print("-" * 60)
            
            # Rerank the results
            print("\n=== Reranking Results ===")
            reranked_results = rerank_images(query, search_results)
            
            print("=== Reranked Results (Top 3) ===")
            for i, result in enumerate(reranked_results[:3], 1):
                print(f"\n{i}. Image: {result['image_id']}")
                print(f"   Vector Score: {result['score']:.3f}")
                print(f"   Rerank Score: {result.get('rerank_score', 'N/A')}")
                print(f"   Page: {result['page']}")
                print(f"   Description: {result['text'][:300]}...")
                print(f"   Path: {result['image_path']}")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def process_and_search():
    """Process images first, then start interactive search"""
    print("=== Image Processing and Search ===")
    
    # Ask if user wants to process images
    process_choice = input("Do you want to process existing images first? (y/n): ").strip().lower()
    
    if process_choice in ['y', 'yes']:
        print("Processing existing images...")
        results = process_existing_images()
        print(f"Processed {len(results)} images.")
    
    # Start interactive search
    interactive_image_search()

if __name__ == "__main__":
    process_and_search()