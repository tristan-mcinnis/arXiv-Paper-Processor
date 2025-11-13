# arXiv Paper Processor

This project allows you to search for academic papers on arXiv, download and process them, and generate responses to specific questions using embeddings and language models. The application leverages several tools including Gradio for the interface, ChromaDB for embedding storage, LangChain for text processing, and LMStudio for local AI inference.

## Features

- **arXiv Paper Search**: Search for papers based on a query, download them in PDF format, and extract relevant metadata.
- **Text Processing**: Extract and split the content of downloaded papers into manageable chunks.
- **Embedding Generation**: Generate text embeddings using the `nomic-embed-text` model and store them in ChromaDB.
- **Document Retrieval**: Retrieve the most relevant document based on a query using the generated embeddings.
- **Response Generation**: Generate responses to specific questions using the LLaMA model via LMStudio, incorporating references to the source documents.

## Requirements

- Python 3.8+
- LMStudio installed and running locally
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

## LMStudio Setup

1. **Download and Install LMStudio**: Download LMStudio from [https://lmstudio.ai/](https://lmstudio.ai/)

2. **Load Required Models**:
   - Open LMStudio and download the following models:
     - `nomic-embed-text` (for embeddings)
     - `llama3.1` (or your preferred LLaMA model for text generation)

3. **Start the Local Server**:
   - In LMStudio, go to the "Local Server" tab
   - Click "Start Server"
   - Ensure the server is running on `http://localhost:1234` (default port)
   - The application uses the OpenAI-compatible API endpoint

4. **Verify Server**: The server should be accessible at `http://localhost:1234/v1`

## Configuration

The application uses a `config.yaml` file for easy customization of all settings. You can modify this file to change models, directories, parameters, and more without editing the code.

### Configuration Options

```yaml
# LMStudio API Configuration
lmstudio:
  base_url: "http://localhost:1234/v1"  # LMStudio server URL
  api_key: "lm-studio"                   # API key (can be any string for local)

# Model Configuration
models:
  embedding: "nomic-embed-text"  # Model for generating embeddings
  chat: "llama3.1"               # Model for text generation

# Generation Settings
generation:
  temperature: 0.7      # Controls randomness (0.0 = deterministic, 1.0 = creative)
  max_tokens: null      # Maximum tokens in response (null = no limit)

# arXiv Search Settings
arxiv:
  max_results: 20       # Maximum number of papers to download
  sort_order: "descending"  # "descending" or "ascending"

# File Storage Settings
storage:
  papers_directory: "arxiv_papers"  # Directory for downloaded PDFs

# Text Processing Settings
text_processing:
  chunk_size: 500      # Size of text chunks for processing
  chunk_overlap: 50    # Overlap between chunks

# Vector Database Settings
chromadb:
  collection_name: "arxiv_papers"  # Name of the ChromaDB collection
  n_results: 1         # Number of relevant documents to retrieve
```

### Customizing the Configuration

To customize the application:
1. Open `config.yaml` in a text editor
2. Modify the values according to your needs
3. Save the file
4. Run the application

**Common Customizations:**
- Change `models.embedding` or `models.chat` to use different models available in LMStudio
- Adjust `generation.temperature` to make responses more creative (higher) or deterministic (lower)
- Increase `arxiv.max_results` to download more papers per search
- Modify `text_processing.chunk_size` for larger or smaller text chunks
- Change `storage.papers_directory` to store PDFs in a different location

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tristan-mcinnis/arXiv-Paper-Processor.git
   cd arXiv-Paper-Processor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start LMStudio Server**:
   - Ensure LMStudio is running with the local server started (see LMStudio Setup above)

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Gradio Interface**:
   - The interface will launch in your web browser
   - Enter your search query and question to start processing papers

## Project Structure

app.py: Main application file containing the core logic.
requirements.txt: List of required Python packages.
README.md: Project documentation.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
