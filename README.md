# arXiv Paper Processor

This project allows you to search for academic papers on arXiv, download and process them, and generate responses to specific questions using embeddings and language models. The application leverages several tools including Gradio for the interface, ChromaDB for embedding storage, LangChain for text processing, and LMStudio for local AI inference.

## Features

### Core Functionality
- **arXiv Paper Search**: Search for papers based on a query, download them in PDF format, and extract relevant metadata
- **Text Processing**: Extract and split the content of downloaded papers into manageable chunks
- **Embedding Generation**: Generate text embeddings using the `nomic-embed-text` model and store them in ChromaDB
- **Document Retrieval**: Retrieve the most relevant documents based on a query using semantic search
- **Response Generation**: Generate responses to specific questions using the LLaMA model via LMStudio with citations

### Quality of Life Features
- **🎨 Enhanced UI**: Beautiful Gradio interface with examples, progress tracking, and helpful information
- **⚡ Smart Caching**: Automatically caches downloaded papers to avoid re-downloading
- **📊 Real-time Progress**: Visual progress bars show what's happening during processing
- **🛡️ Error Handling**: Comprehensive error handling with user-friendly error messages
- **📝 Logging**: Detailed logging to `arxiv_processor.log` for debugging and monitoring
- **✅ Input Validation**: Validates user inputs before processing to prevent errors
- **🔄 Batch Processing**: Processes embeddings in batches for improved performance
- **🔒 Environment Variables**: Support for `.env` file to override configuration securely

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

### Environment Variables (Optional)

You can override configuration settings using environment variables by creating a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your values
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
```

Environment variables take precedence over `config.yaml` settings.

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
   - The interface will launch in your web browser (usually `http://localhost:7860`)
   - Enter your search query and question to start processing papers
   - Try the example queries to get started quickly

## Advanced Features

### Smart Caching

The application automatically caches downloaded papers:
- Papers are stored in the `arxiv_papers` directory (configurable)
- Already downloaded papers are detected and reused
- Saves time and bandwidth on repeated queries

### Logging

Comprehensive logging is available:
- All operations are logged to `arxiv_processor.log`
- Console output shows real-time progress
- Helps with debugging and monitoring application behavior

### Error Handling

Robust error handling throughout:
- User-friendly error messages in the UI
- Validates inputs before processing
- Gracefully handles API failures, network issues, and missing files
- Provides suggestions for common problems (e.g., "Is LMStudio running?")

### Progress Tracking

Real-time progress updates show:
- Current operation (searching, downloading, processing)
- Number of papers downloaded vs. cached
- Embedding generation progress
- Estimated completion

## Troubleshooting

### LMStudio Connection Issues

If you get errors about LMStudio not being available:
1. Check that LMStudio is running
2. Verify the local server is started in LMStudio
3. Ensure the server is accessible at `http://localhost:1234`
4. Check that your models are loaded in LMStudio

### No Papers Found

If no papers are downloaded:
1. Try a more general search query
2. Check your internet connection
3. Verify arXiv is accessible
4. Check the logs in `arxiv_processor.log`

### ChromaDB Errors

If you encounter ChromaDB issues:
1. The application now uses `get_or_create_collection` to avoid conflicts
2. If problems persist, delete the `chroma/` directory to reset the database

### Configuration Errors

If the application won't start:
1. Verify `config.yaml` exists and is valid YAML
2. Check that all required sections are present
3. Review `arxiv_processor.log` for specific error messages

## Project Structure

```
arXiv-Paper-Processor/
├── app.py                  # Main application file with enhanced features
├── config.yaml             # Configuration file for all settings
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables file
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── arxiv_papers/          # Downloaded PDFs (auto-created, gitignored)
├── arxiv_processor.log    # Application logs (auto-created, gitignored)
└── chroma/                # ChromaDB data (auto-created, gitignored)
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Changelog

### Version 2.0 (Latest)
- ✨ Enhanced Gradio UI with examples and better organization
- 🚀 Added real-time progress tracking
- 🛡️ Comprehensive error handling and validation
- 📝 Logging system for debugging
- ⚡ Smart caching for downloaded papers
- 🔄 Batch embedding generation for better performance
- 🔒 Environment variable support
- ✅ Input validation and user-friendly error messages
- 🎨 Better UI feedback and status updates

### Version 1.0
- 🔄 Migration from Ollama to LMStudio
- ⚙️ Configuration via `config.yaml`
- 📚 Basic arXiv paper processing and Q&A functionality
