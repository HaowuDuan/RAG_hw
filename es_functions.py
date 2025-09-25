from config import get_es

def create_elastic_index(index_name):
    es=get_es()
    mappings = {
                "properties": {
                    "text": {
                        "type": "text"
                    }, 
                    "vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                        }
                    #metadata filtering
                    # "file_name": {
                    #     "type": "text"
                    # },
                    # "page": {
                    #     "type": "integer"
                    # },
                    # "chapter": {
                    #     "type": "text"
                    # },
                    # "doc_type": {
                    #     "type": "text"
                    # },
                    # "language": {
                    #     "type": "text"
                    # },
                    # "file_id": {
                    #      "type": "long"
                    # },  

                    }
                }
    #创建elastic
    try:
        es.indices.create(index=index_name, mappings=mappings)
        print('[Create Vector DB]' + index_name + ' created')
    except Exception as e:
        print(f'Create Vector DB Exception: {e}')

def delete_elastic_index(index_name):
    es=get_es()
    es.indices.delete(index=index_name)
    print('[Delete Vector DB]' + index_name + ' deleted')

if __name__ == '__main__':
    # Delete and recreate the index
    #delete_elastic_index('hw_index')
    create_elastic_index('hw_index')