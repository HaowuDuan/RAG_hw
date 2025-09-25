from config import get_es
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import local_embedding
import time
import tiktoken
import fitz  # PyMuPDF
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_image_description(image_data, page_context=""):
    """Generate description for image using LLM with context"""
    try:
        # Convert to base64 for API
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables")
            return "Figure or chart (API key not configured)"
        
        # Create context-aware prompt
        if page_context:
            prompt = f"""Describe this figure/chart/diagram in detail, focusing on key information that would be useful for search and retrieval.

Context from the surrounding page:
{page_context[:1000]}

Please consider how this image relates to the surrounding text and include any relevant captions or references."""
        else:
            prompt = "Describe this figure/chart/diagram in detail, focusing on key information that would be useful for search and retrieval."
        
        # Use current OpenAI API format
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # smaller, more cost-effective vision model
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating image description: {e}")
        return "Figure or chart (description generation failed)"

def process_pdf(es_index, file_path):
    es = get_es()
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    
    ################# 用Cluster生成文件摘要 ################
    # try:
    #     file_summary = generate_summary_for_file(subtitle, pages, file_id, None, user_id, base_id)
    # except Exception as e:
    #     pass

    ################# 处理文本内容 ################
    textsplit = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, length_function=num_tokens_from_string)
    chunks = textsplit.split_documents(pages)
    batch = []
    
    for i, chunk in enumerate(chunks):
        batch.append(chunk)

        if len(batch) == 25 or i == len(chunks) - 1: 
            embeddings = local_embedding([b.page_content for b in batch])
            for j, pc in enumerate(batch):
                body = {
                    'text': pc.page_content,
                    'vector': embeddings[j],
                    'type': 'text',
                    'page': pc.metadata.get('page', 0),
                    'source_pdf': file_path
                }
                retry = 0
                while retry <= 5:
                    try:
                        es.index(index=es_index, body=body)
                        break
                    except Exception as e:
                        print(f'[Elastic Error] {str(e)} retry')
                        retry += 1
                        time.sleep(1)
            batch = []

    ################# 提取图片和表格 ################
    print("Processing images from PDF...")
    doc = fitz.open(file_path)
    
    # Create images directory if it doesn't exist
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    image_count = 0
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract page context for better image descriptions
        page_text = page.get_text("dict", sort=True)
        page_context = ""
        for block in page_text['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        page_context += span['text'] + " "
        
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # Not CMYK
                    img_data = pix.tobytes("png")
                    
                    # Generate unique image ID
                    pdf_name = os.path.basename(file_path).replace('.pdf', '')
                    image_id = f"{pdf_name}_page{page_num+1}_img{img_index}"
                    image_path = os.path.join(images_dir, f"{image_id}.png")
                    
                    # Save image to disk
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                    
                    # Generate description using LLM with context
                    print(f"Generating description for image {image_count + 1}...")
                    description = generate_image_description(img_data, page_context)
                    
                    # Create embedding for the description using local model
                    img_embedding = local_embedding([description])[0]
                    if img_embedding is None:
                        print(f"Failed to get embedding for image {image_id}, skipping")
                        pix = None  # Free memory before continuing
                        continue
                    
                    # Index the image with its description
                    body = {
                        'text': description,
                        'vector': img_embedding,
                        'type': 'image',
                        'page': page_num + 1,
                        'image_id': image_id,
                        'image_path': image_path,
                        'source_pdf': file_path
                    }
                    
                    retry = 0
                    while retry <= 5:
                        try:
                            es.index(index=es_index, body=body)
                            print(f"Indexed image: {image_id}")
                            break
                        except Exception as e:
                            print(f'[Elastic Error] {str(e)} retry')
                            retry += 1
                            time.sleep(1)
                    
                    image_count += 1
                
                # Free memory
                pix = None
                
            except Exception as e:
                print(f"Error processing image on page {page_num+1}: {e}")
                continue
    
    doc.close()
    print(f"Processed {image_count} images from {file_path}")

def num_tokens_from_string(string):   
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    process_pdf('hw_index', 'LLM_fintuing_guide.pdf')