from elasticsearch import Elasticsearch
import time

class ElasticConfig:
    url='http://elastic:nsKZDPoN@localhost:9200'  # Updated with correct credentials
    
    
def get_es():
    while True:
        try:    
            es = Elasticsearch([ElasticConfig.url]) 
            return es
        except:
            print('ElasticSearch conn failed, retry')
            time.sleep(3)
            
EMBEDDING_URL="http://test.2brain.cn:9800/v1/emb"
RERANK_URL="http://test.2brain.cn:2260/rerank"
IMAGE_MODEL_URL='http://test.2brain.cn:23333/v1'
