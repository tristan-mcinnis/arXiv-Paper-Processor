import gradio as gr
import os
import time
import arxiv
from openai import OpenAI
import chromadb
import yaml
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load configuration from config.yaml
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize LMStudio client (OpenAI-compatible API)
lm_client = OpenAI(
    base_url=config['lmstudio']['base_url'],
    api_key=config['lmstudio']['api_key']
)

def process_papers(query, question_text):
    dirpath = config['storage']['papers_directory']
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    print("Starting arXiv search...")
    client = arxiv.Client()

    # Determine sort order from config
    sort_order = arxiv.SortOrder.Descending if config['arxiv']['sort_order'].lower() == 'descending' else arxiv.SortOrder.Ascending

    search = arxiv.Search(
        query=query,
        max_results=config['arxiv']['max_results'],
        sort_order=sort_order
    )

    papers_metadata = []
    for result in client.results(search):
        metadata = {
            'title': result.title,
            'authors': ', '.join([author.name for author in result.authors]),
            'url': result.pdf_url,
            'date': result.published.date().strftime('%Y-%m-%d')  # Convert date to string
        }
        try:
            print(f"Downloading paper: {result.title}")
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            papers_metadata.append(metadata)
        except (FileNotFoundError, ConnectionResetError) as e:
            print("Error occurred:", e)
            time.sleep(5)

    print("Loading papers...")
    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    papers = []
    try:
        papers = loader.load()
        print(f"{len(papers)} papers loaded.")
    except Exception as e:
        print(f"Error loading file: {e}")

    full_text = ''
    for paper in papers:
        full_text += paper.page_content

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['text_processing']['chunk_size'],
        chunk_overlap=config['text_processing']['chunk_overlap']
    )
    paper_chunks = text_splitter.create_documents([full_text])
    print(f"Text split into {len(paper_chunks)} chunks.")

    print("Initializing ChromaDB...")
    client = chromadb.Client()
    collection = client.create_collection(name=config['chromadb']['collection_name'])

    print("Generating embeddings and storing in ChromaDB...")
    for i, chunk in enumerate(paper_chunks):
        response = lm_client.embeddings.create(
            model=config['models']['embedding'],
            input=chunk.page_content
        )
        embedding = response.data[0].embedding

        # Ensure metadata is properly linked with text chunks
        chunk_metadata = papers_metadata[min(i, len(papers_metadata) - 1)]

        # Ensure metadata dictionary is not empty or missing required fields
        if not all(key in chunk_metadata for key in ['title', 'authors', 'url', 'date']):
            chunk_metadata = {
                'title': 'Unknown Title',
                'authors': 'Unknown Authors',
                'url': 'Unknown URL',
                'date': 'Unknown Date'
            }

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[chunk_metadata]
        )
        print(f"Stored chunk {i+1} of {len(paper_chunks)}")

    print("Generating embedding for the query and retrieving the most relevant document...")
    response = lm_client.embeddings.create(
        model=config['models']['embedding'],
        input=question_text
    )
    results = collection.query(
        query_embeddings=[response.data[0].embedding],
        n_results=config['chromadb']['n_results']
    )
    
    if results and results['documents']:
        context_data = results['documents'][0][0]
        context_metadata_list = results['metadatas'][0]  # Get the first metadata entry from the list
        print("Relevant document found.")
    else:
        context_data = "No relevant documents found."
        context_metadata_list = [{
            'title': 'Unknown Title',
            'authors': 'Unknown Authors',
            'url': 'Unknown URL',
            'date': 'Unknown Date'
        }]
        print("No relevant documents found.")

    # Generate references from metadata
    references = []
    for context_metadata in context_metadata_list:
        reference = f"{context_metadata.get('title', 'Unknown Title')} by {context_metadata.get('authors', 'Unknown Authors')} ({context_metadata.get('date', 'Unknown Date')}) - [PDF]({context_metadata.get('url', 'Unknown URL')})"
        references.append(reference)

    references_text = "\n".join(references)

    # Use the configured chat model for generating the final response
    prompt_text = f"Using this data: {context_data}. Respond to this prompt: {question_text}"

    # Build chat completion parameters
    completion_params = {
        "model": config['models']['chat'],
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": config['generation']['temperature']
    }

    # Add max_tokens if specified in config
    if config['generation']['max_tokens'] is not None:
        completion_params["max_tokens"] = config['generation']['max_tokens']

    completion = lm_client.chat.completions.create(**completion_params)
    final_response = completion.choices[0].message.content

    final_output = f"{final_response}\n\n**References:**\n{references_text}"

    print("Final response generated.")
    return final_output

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_papers,
    inputs=["text", "text"],
    outputs="text",
    description="Enter a search query and a question to process arXiv papers, with sources referenced."
)

iface.launch()
