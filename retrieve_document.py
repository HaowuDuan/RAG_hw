import re
import requests
from config import get_es, RERANK_URL
from embedding import local_embedding
import jieba

def elastic_search(text, es_index):
    es=get_es()
    key_words = get_keyword(text)


    keyword_query = {
        "bool": {
            "should": [
                {"match": {"text": {"query": keyword, "fuzziness": "AUTO"}}} for keyword in key_words
            ],
            "minimum_should_match": 1
        }
    }
    res_keyword = es.search(index=es_index, query=keyword_query)
    keyword_hits = [{'id': hit['_id'], 'text': hit['_source'].get('text'), 
                     'file_id': hit['_source'].get('file_id'), 'image_id': hit['_source'].get('image_id'), 'metadata':hit['_source'].get('metadata'),
                     'rank': idx + 1} for idx, hit in enumerate(res_keyword['hits']['hits'])]
    # print(keyword_hits)
    # keyword_hits = [] #test vector search

    embedding = local_embedding([text])
    vector_query = {
        "bool": {
            "must": [{"match_all": {}}],
            "should": [
                {"script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                        "params": {"queryVector": embedding[0]}
                    }
                }}
            ]
        }
    }
    res_vector = es.search(index=es_index, query=vector_query)
    
    # print(res_vector)
    
    vector_hits = [{'id': hit['_id'], 'text': hit['_source'].get('text'), 
                    'file_id': hit['_source'].get('file_id'),'image_id': hit['_source'].get('image_id'), 'metadata':hit['_source'].get('metadata'),
                    'rank': idx + 1} for idx, hit in enumerate(res_vector['hits']['hits'])]
    
    # print(vector_hits)
    combined_results = hybrid_search_rrf(keyword_hits, vector_hits)
    # print(combined_results)
    return combined_results

def get_keyword(query):
    # Ensure input is string type
    if not isinstance(query, str):
        print(f'[Get Keyword] Received non-string query: {query} (type: {type(query)}), converting to string')
        query = str(query) if query is not None else ''
    
    # Ensure query is not empty
    if not query.strip():
        print('[Get Keyword] Empty query string, returning empty list')
        return []
    
    try:
        # Use search engine mode for word segmentation
        seg_list = jieba.cut_for_search(query)
        # Filter out stop words
        filtered_keywords = [word for word in seg_list if word not in stop_words]
        # logging.info('[Jieba Keywords Extraction] ' + ','.join(filtered_keywords))
        return filtered_keywords
    except Exception as e:
        print(f'[Get Keyword] Error processing query "{query}": {e}')
        return []
    
stop_words = set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    "i", "you", "we", "they", "she", "him", "her", "his", "their", "them", "this", "these", "those", "my", "your", "our", "me", "us", "himself", "herself", "itself",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would", "could", "should", "may", "might", "must", "can",
    "but", "or", "nor", "so", "yet", "because", "if", "unless", "although", "though", "while", "where", "when", "why", "how", "what", "which", "who", "whom", "whose",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "now",
    "about", "above", "after", "again", "against", "all", "also", "and", "any", "are", "as", "at", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
])
    
    
def hybrid_search_rrf(keyword_hits, vector_hits, k=60):
    # Initialize score dictionary
    scores = {}
    
    # Process keyword hits
    for hit in keyword_hits:
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {'score': 0, 'text': hit['text'], 'id': doc_id, 'file_id': hit['file_id'],'image_id':hit['image_id'],'metadata':hit['metadata']}
        scores[doc_id]['score'] += 1 / (k + hit['rank'])
    
    # Process vector hits
    for hit in vector_hits:
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {'score': 0, 'text': hit['text'], 'id': doc_id, 'file_id': hit['file_id'],'image_id':hit['image_id'],'metadata':hit['metadata']}
        scores[doc_id]['score'] += 1 / (k + hit['rank'])
    
    # Sort documents by their RRF score and assign ranks
    ranked_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)

    # Removing the timestamps
    for _, doc in enumerate(ranked_docs):
        timestamp_pattern = re.compile(r'\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}')
        doc['text'] = re.sub(timestamp_pattern, '', doc['text'])    
        
    # Format the final list of results                      # Remove image tags
    final_results = [{'id': doc['id'], 'text': doc['text'], 'file_id': doc['file_id'],'image_id':doc['image_id'], 'metadata':doc['metadata'],'rank': idx + 1} for idx, doc in enumerate(ranked_docs)]
    # print(final_results)
    return final_results

def rerank(query, result_doc):

    res = requests.post(RERANK_URL, json={"query": query, "documents": [doc['text'] for doc in result_doc]}).json()
    if res and 'scores' in res and len(res['scores']) == len(result_doc):
        for idx, doc in enumerate(result_doc):
            result_doc[idx]['score'] = res['scores'][idx]
        
        # Sort documents by rerank score in descending order (highest scores first)
        result_doc.sort(key=lambda x: x['score'], reverse=True)
            
    return result_doc

def interactive_search():
    """Interactive search interface"""
    print("=== Text Search Interface ===")
    print("Enter your search query (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nSearch query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                print("Please enter a search query.")
                continue
            
            print(f"\nSearching for: '{query}'")
            print("=" * 60)
            
            # Perform search
            result_doc = elastic_search(query, 'hw_index')
            
            if not result_doc:
                print("No results found.")
                continue
            
            print("=== Initial Retrieval Results ===")
            for idx, doc in enumerate(result_doc[:5], 1):  # Show top 5
                doc_type = doc.get('type', 'text')
                print(f"{idx}. [{doc_type.upper()}] {doc['text'][:150]}{'...' if len(doc['text']) > 150 else ''}")
                if doc_type == 'image':
                    print(f"   Image: {doc.get('image_id', 'N/A')} | Page: {doc.get('page', 'N/A')}")
                print()
            
            # Rerank results
            reranked_results = rerank(query, result_doc)
            print("=== Reranked Results (Top 3) ===")
            for idx, doc in enumerate(reranked_results[:3], 1):
                doc_type = doc.get('type', 'text')
                print(f"{idx}. [{doc_type.upper()}] Score: {doc.get('score', '-')}")
                print(f"   {doc['text'][:200]}{'...' if len(doc['text']) > 200 else ''}")
                if doc_type == 'image':
                    print(f"   Image: {doc.get('image_id', 'N/A')} | Page: {doc.get('page', 'N/A')}")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    interactive_search()
