import gradio as gr
import os
import time
import arxiv
from openai import OpenAI
import chromadb
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
def load_config():
    """Load and validate configuration from config.yaml"""
    try:
        if not os.path.exists('config.yaml'):
            raise FileNotFoundError("config.yaml not found. Please create it from the template.")

        with open('config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Validate required configuration sections
        required_sections = ['provider', 'generation', 'arxiv', 'storage', 'text_processing', 'chromadb']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate provider sections exist
        supported_providers = ['lmstudio', 'openai', 'kimi', 'deepseek']
        for provider_type in ['embedding', 'chat']:
            provider_name = config['provider'][provider_type]
            if provider_name not in supported_providers:
                raise ValueError(f"Unsupported provider '{provider_name}' for {provider_type}")
            if provider_name not in config:
                raise ValueError(f"Configuration for provider '{provider_name}' not found")

        # Allow environment variables to override config for each provider
        # LMStudio
        if 'lmstudio' in config:
            config['lmstudio']['base_url'] = os.getenv('LMSTUDIO_BASE_URL', config['lmstudio']['base_url'])
            config['lmstudio']['api_key'] = os.getenv('LMSTUDIO_API_KEY', config['lmstudio']['api_key'])

        # OpenAI
        if 'openai' in config:
            config['openai']['base_url'] = os.getenv('OPENAI_BASE_URL', config['openai']['base_url'])
            config['openai']['api_key'] = os.getenv('OPENAI_API_KEY', config['openai']['api_key'])

        # Kimi
        if 'kimi' in config:
            config['kimi']['base_url'] = os.getenv('KIMI_BASE_URL', config['kimi']['base_url'])
            config['kimi']['api_key'] = os.getenv('KIMI_API_KEY', config['kimi']['api_key'])

        # DeepSeek
        if 'deepseek' in config:
            config['deepseek']['base_url'] = os.getenv('DEEPSEEK_BASE_URL', config['deepseek']['base_url'])
            config['deepseek']['api_key'] = os.getenv('DEEPSEEK_API_KEY', config['deepseek']['api_key'])

        logger.info("Configuration loaded successfully")
        logger.info(f"Using {config['provider']['embedding']} for embeddings and {config['provider']['chat']} for chat")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

config = load_config()

# Initialize AI clients
def get_client(provider_name: str):
    """
    Get an OpenAI-compatible client for the specified provider

    Args:
        provider_name: Name of the provider (lmstudio, openai, kimi, deepseek)

    Returns:
        OpenAI client configured for the provider
    """
    try:
        provider_config = config[provider_name]
        base_url = provider_config['base_url']
        api_key = provider_config['api_key']

        if not api_key and provider_name != 'lmstudio':
            logger.warning(f"No API key configured for {provider_name}. Set it in config.yaml or use environment variables.")

        client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "dummy-key"  # Some providers need a key even if not used
        )

        logger.info(f"Initialized client for {provider_name} at {base_url}")
        return client

    except Exception as e:
        logger.error(f"Error initializing {provider_name} client: {e}")
        raise

def get_model_name(provider_name: str, model_type: str) -> str:
    """
    Get the model name for a specific provider and model type

    Args:
        provider_name: Name of the provider
        model_type: Type of model (embedding or chat)

    Returns:
        Model name string
    """
    return config[provider_name]['models'][model_type]

# Initialize clients based on configuration
embedding_provider = config['provider']['embedding']
chat_provider = config['provider']['chat']

embedding_client = get_client(embedding_provider)
chat_client = get_client(chat_provider)

logger.info(f"Embedding provider: {embedding_provider} with model {get_model_name(embedding_provider, 'embedding')}")
logger.info(f"Chat provider: {chat_provider} with model {get_model_name(chat_provider, 'chat')}")

def is_paper_downloaded(paper_id: str, dirpath: str) -> bool:
    """Check if a paper is already downloaded"""
    pdf_path = Path(dirpath) / f"{paper_id}.pdf"
    return pdf_path.exists()

def process_papers(query, question_text, progress=gr.Progress()):
    """
    Process arXiv papers and answer questions using LMStudio

    Args:
        query: arXiv search query
        question_text: Question to answer about the papers
        progress: Gradio progress tracker

    Returns:
        Final response with references or error message
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "❌ Error: Please enter a valid search query."

        if not question_text or not question_text.strip():
            return "❌ Error: Please enter a question."

        if len(query.strip()) < 3:
            return "❌ Error: Search query must be at least 3 characters long."

        logger.info(f"Starting paper processing - Query: '{query}', Question: '{question_text}'")

        # Setup directory
        dirpath = config['storage']['papers_directory']
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        # Step 1: Search and download papers
        progress(0.1, desc="Searching arXiv...")
        logger.info("Starting arXiv search...")

        try:
            client = arxiv.Client()
            sort_order = arxiv.SortOrder.Descending if config['arxiv']['sort_order'].lower() == 'descending' else arxiv.SortOrder.Ascending

            search = arxiv.Search(
                query=query,
                max_results=config['arxiv']['max_results'],
                sort_order=sort_order
            )
        except Exception as e:
            logger.error(f"Error creating arXiv search: {e}")
            return f"❌ Error setting up arXiv search: {str(e)}"

        papers_metadata = []
        downloaded_count = 0
        cached_count = 0

        progress(0.2, desc="Downloading papers...")

        try:
            for idx, result in enumerate(client.results(search)):
                paper_id = result.get_short_id()
                metadata = {
                    'title': result.title,
                    'authors': ', '.join([author.name for author in result.authors]),
                    'url': result.pdf_url,
                    'date': result.published.date().strftime('%Y-%m-%d')
                }

                try:
                    # Check if paper is already downloaded (smart caching)
                    if is_paper_downloaded(paper_id, dirpath):
                        logger.info(f"Paper already downloaded: {result.title}")
                        papers_metadata.append(metadata)
                        cached_count += 1
                    else:
                        logger.info(f"Downloading paper: {result.title}")
                        result.download_pdf(dirpath=dirpath)
                        logger.info(f"Downloaded paper id {paper_id}")
                        papers_metadata.append(metadata)
                        downloaded_count += 1

                except (FileNotFoundError, ConnectionResetError) as e:
                    logger.warning(f"Error downloading paper '{result.title}': {e}")
                    time.sleep(5)
                except Exception as e:
                    logger.warning(f"Unexpected error downloading paper: {e}")

                # Update progress
                progress(0.2 + (0.2 * (idx + 1) / config['arxiv']['max_results']),
                        desc=f"Downloaded {downloaded_count}, cached {cached_count} papers...")

        except Exception as e:
            logger.error(f"Error during paper download: {e}")
            return f"❌ Error downloading papers: {str(e)}"

        if not papers_metadata:
            return "❌ No papers were successfully downloaded. Please try a different query."

        logger.info(f"Downloaded {downloaded_count} new papers, used {cached_count} cached papers")

        # Step 2: Load papers
        progress(0.4, desc="Loading PDF content...")
        logger.info("Loading papers...")

        try:
            loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
            papers = loader.load()
            logger.info(f"{len(papers)} paper pages loaded.")

            if not papers:
                return "❌ No papers could be loaded. The PDFs might be corrupted."

        except Exception as e:
            logger.error(f"Error loading papers: {e}")
            return f"❌ Error loading PDF files: {str(e)}"

        # Step 3: Process text
        progress(0.5, desc="Processing text...")

        full_text = ''.join(paper.page_content for paper in papers)

        if not full_text.strip():
            return "❌ No text could be extracted from the papers."

        logger.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['text_processing']['chunk_size'],
            chunk_overlap=config['text_processing']['chunk_overlap']
        )
        paper_chunks = text_splitter.create_documents([full_text])
        logger.info(f"Text split into {len(paper_chunks)} chunks.")

        # Step 4: Initialize ChromaDB
        progress(0.55, desc="Initializing vector database...")
        logger.info("Initializing ChromaDB...")

        try:
            chroma_client = chromadb.Client()
            # Use get_or_create_collection to prevent errors if collection exists
            collection = chroma_client.get_or_create_collection(
                name=config['chromadb']['collection_name']
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            return f"❌ Error initializing database: {str(e)}"

        # Step 5: Generate embeddings
        progress(0.6, desc="Generating embeddings...")
        logger.info("Generating embeddings and storing in ChromaDB...")

        try:
            # Batch embeddings for better performance
            batch_size = 10
            embedding_model = get_model_name(embedding_provider, 'embedding')

            for i in range(0, len(paper_chunks), batch_size):
                batch = paper_chunks[i:i + batch_size]
                batch_texts = [chunk.page_content for chunk in batch]

                # Generate embeddings for batch
                response = embedding_client.embeddings.create(
                    model=embedding_model,
                    input=batch_texts
                )

                # Store each embedding
                for j, chunk in enumerate(batch):
                    chunk_idx = i + j
                    embedding = response.data[j].embedding

                    # Link metadata
                    chunk_metadata = papers_metadata[min(chunk_idx, len(papers_metadata) - 1)]

                    # Ensure metadata has required fields
                    if not all(key in chunk_metadata for key in ['title', 'authors', 'url', 'date']):
                        chunk_metadata = {
                            'title': 'Unknown Title',
                            'authors': 'Unknown Authors',
                            'url': 'Unknown URL',
                            'date': 'Unknown Date'
                        }

                    collection.add(
                        ids=[str(chunk_idx)],
                        embeddings=[embedding],
                        documents=[chunk.page_content],
                        metadatas=[chunk_metadata]
                    )

                # Update progress
                progress_val = 0.6 + (0.2 * (i + batch_size) / len(paper_chunks))
                progress(min(progress_val, 0.8), desc=f"Stored {min(i + batch_size, len(paper_chunks))} / {len(paper_chunks)} chunks...")

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return f"❌ Error generating embeddings with {embedding_provider}. Check your API key and connection. Error: {str(e)}"

        # Step 6: Query for relevant documents
        progress(0.85, desc="Finding relevant information...")
        logger.info("Generating embedding for the query and retrieving relevant documents...")

        try:
            embedding_model = get_model_name(embedding_provider, 'embedding')
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=question_text
            )
            results = collection.query(
                query_embeddings=[response.data[0].embedding],
                n_results=config['chromadb']['n_results']
            )
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return f"❌ Error searching documents with {embedding_provider}: {str(e)}"

        if results and results['documents'] and results['documents'][0]:
            context_data = results['documents'][0][0]
            context_metadata_list = results['metadatas'][0]
            logger.info("Relevant documents found.")
        else:
            context_data = "No relevant documents found."
            context_metadata_list = [{
                'title': 'Unknown Title',
                'authors': 'Unknown Authors',
                'url': 'Unknown URL',
                'date': 'Unknown Date'
            }]
            logger.warning("No relevant documents found.")

        # Generate references
        references = []
        for context_metadata in context_metadata_list:
            reference = f"{context_metadata.get('title', 'Unknown Title')} by {context_metadata.get('authors', 'Unknown Authors')} ({context_metadata.get('date', 'Unknown Date')}) - [PDF]({context_metadata.get('url', 'Unknown URL')})"
            references.append(reference)
        references_text = "\n".join(references)

        # Step 7: Generate final response
        progress(0.9, desc="Generating response...")
        logger.info("Generating final response with LLM...")

        try:
            prompt_text = f"Using this data: {context_data}. Respond to this prompt: {question_text}"

            chat_model = get_model_name(chat_provider, 'chat')
            completion_params = {
                "model": chat_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": config['generation']['temperature']
            }

            if config['generation']['max_tokens'] is not None:
                completion_params["max_tokens"] = config['generation']['max_tokens']

            completion = chat_client.chat.completions.create(**completion_params)
            final_response = completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error generating response with {chat_provider}. Check your API key and connection. Error: {str(e)}"

        final_output = f"{final_response}\n\n**References:**\n{references_text}"

        progress(1.0, desc="Complete!")
        logger.info("Processing completed successfully")
        return final_output

    except Exception as e:
        logger.error(f"Unexpected error in process_papers: {e}", exc_info=True)
        return f"❌ An unexpected error occurred: {str(e)}\n\nPlease check the logs for more details."

# Set up the Gradio interface
with gr.Blocks(title="arXiv Paper Processor", theme=gr.themes.Soft()) as iface:
    gr.Markdown("""
    # 📚 arXiv Paper Processor

    Search for academic papers on arXiv, download and process them, then ask questions about the content.
    The system uses LMStudio for embeddings and text generation with local AI models.
    """)

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="📖 arXiv Search Query",
                placeholder="e.g., 'machine learning transformers' or 'quantum computing algorithms'",
                lines=2,
                info=f"Enter keywords to search arXiv (will retrieve up to {config['arxiv']['max_results']} papers)"
            )

            question_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="e.g., 'What are the main applications of transformers in NLP?'",
                lines=3,
                info="Ask a specific question about the papers that will be found"
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Process Papers", variant="primary", size="lg")
                clear_btn = gr.ClearButton([query_input, question_input], value="🗑️ Clear")

        with gr.Column():
            output = gr.Textbox(
                label="📝 Response",
                lines=20,
                show_copy_button=True,
                info="AI-generated answer with references to source papers"
            )

    # Example queries
    gr.Markdown("### 💡 Example Queries")
    gr.Examples(
        examples=[
            ["attention is all you need", "What is the transformer architecture and how does it work?"],
            ["quantum computing error correction", "What are the main challenges in quantum error correction?"],
            ["reinforcement learning robotics", "How is reinforcement learning applied to robot control?"],
            ["generative adversarial networks", "What are GANs and what are their applications?"],
            ["graph neural networks", "How do graph neural networks differ from traditional neural networks?"]
        ],
        inputs=[query_input, question_input],
        label="Click an example to try it out"
    )

    # Info section
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
        1. **Search**: Searches arXiv for papers matching your query
        2. **Download**: Downloads relevant papers (with smart caching)
        3. **Process**: Extracts and chunks text from PDFs
        4. **Embed**: Generates embeddings using your configured model
        5. **Query**: Finds most relevant content for your question
        6. **Generate**: Uses LLM to create a comprehensive answer with citations

        **Note**: Make sure your AI provider is configured correctly. If using LMStudio, ensure it's running. For cloud providers (OpenAI, Kimi, DeepSeek), ensure your API keys are set.
        """)

    with gr.Accordion("⚙️ Current Configuration", open=False):
        gr.Markdown(f"""
        **AI Providers:**
        - **Embedding Provider**: `{embedding_provider}` using model `{get_model_name(embedding_provider, 'embedding')}`
        - **Chat Provider**: `{chat_provider}` using model `{get_model_name(chat_provider, 'chat')}`

        **Processing Settings:**
        - **Max Papers**: {config['arxiv']['max_results']}
        - **Chunk Size**: {config['text_processing']['chunk_size']}
        - **Temperature**: {config['generation']['temperature']}
        - **Storage Directory**: `{config['storage']['papers_directory']}`

        Edit `config.yaml` to change these settings or switch providers.
        """)

    submit_btn.click(
        fn=process_papers,
        inputs=[query_input, question_input],
        outputs=output
    )

if __name__ == "__main__":
    logger.info("Starting arXiv Paper Processor Gradio interface...")
    iface.launch()
