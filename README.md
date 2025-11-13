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

## AI Provider Setup

The application supports multiple AI providers for flexibility. Choose the one that best fits your needs:

### Supported Providers

1. **LMStudio** (Local, Free) - Run AI models locally on your machine
2. **OpenAI** (Cloud, Paid) - GPT-4, GPT-3.5, and OpenAI embeddings
3. **Kimi** (Cloud, Paid) - Moonshot AI's models with excellent Chinese support
4. **DeepSeek** (Cloud, Paid) - Competitive pricing with strong performance

### LMStudio Setup (Local AI - Free)

1. **Download and Install LMStudio**: [https://lmstudio.ai/](https://lmstudio.ai/)

2. **Load Required Models**:
   - Open LMStudio and download models:
     - `nomic-embed-text` (for embeddings)
     - `llama3.1` or similar (for text generation)

3. **Start the Local Server**:
   - In LMStudio, go to the "Local Server" tab
   - Click "Start Server"
   - Ensure the server is running on `http://localhost:1234`

4. **Configure in config.yaml**:
   ```yaml
   provider:
     embedding: "lmstudio"
     chat: "lmstudio"
   ```

### OpenAI Setup (Cloud API)

1. **Get API Key**: Sign up at [https://platform.openai.com/](https://platform.openai.com/)

2. **Add API Key**:
   - Create a `.env` file: `cp .env.example .env`
   - Add your key: `OPENAI_API_KEY=your-key-here`
   - Or add directly to `config.yaml`

3. **Configure in config.yaml**:
   ```yaml
   provider:
     embedding: "openai"
     chat: "openai"

   openai:
     api_key: "your-key-here"  # Or use .env
     models:
       embedding: "text-embedding-3-small"
       chat: "gpt-4o-mini"  # or gpt-4o, gpt-4-turbo
   ```

### Kimi (Moonshot AI) Setup

1. **Get API Key**: Register at [https://platform.moonshot.cn/](https://platform.moonshot.cn/)

2. **Add API Key**:
   - Add to `.env`: `KIMI_API_KEY=your-key-here`
   - Or add to `config.yaml`

3. **Configure in config.yaml**:
   ```yaml
   provider:
     embedding: "kimi"
     chat: "kimi"

   kimi:
     api_key: "your-key-here"
     models:
       embedding: "moonshot-v1-8k"
       chat: "moonshot-v1-8k"  # or moonshot-v1-32k, moonshot-v1-128k
   ```

### DeepSeek Setup

1. **Get API Key**: Register at [https://platform.deepseek.com/](https://platform.deepseek.com/)

2. **Add API Key**:
   - Add to `.env`: `DEEPSEEK_API_KEY=your-key-here`
   - Or add to `config.yaml`

3. **Configure in config.yaml**:
   ```yaml
   provider:
     embedding: "deepseek"
     chat: "deepseek"

   deepseek:
     api_key: "your-key-here"
     models:
       embedding: "deepseek-chat"
       chat: "deepseek-chat"  # or deepseek-coder
   ```

### Mix and Match Providers

You can use different providers for embeddings and chat:

```yaml
provider:
  embedding: "openai"      # Use OpenAI for embeddings
  chat: "lmstudio"         # Use LMStudio for chat (save costs!)
```

## Configuration

The application uses a `config.yaml` file for easy customization. You can switch between providers, change models, and adjust parameters without editing code.

### Key Configuration Sections

#### Provider Selection
```yaml
provider:
  embedding: "lmstudio"  # Which provider for embeddings
  chat: "lmstudio"       # Which provider for chat
```

#### Processing Settings
```yaml
# Generation Settings
generation:
  temperature: 0.7      # Controls randomness (0.0 = deterministic, 1.0 = creative)
  max_tokens: null      # Maximum tokens in response (null = no limit)

# arXiv Search Settings
arxiv:
  max_results: 20       # Maximum number of papers to download
  sort_order: "descending"  # "descending" or "ascending"

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
- Change `provider.embedding` or `provider.chat` to switch AI providers
- Adjust model names in each provider's config for different capabilities
- Adjust `generation.temperature` to make responses more creative (higher) or deterministic (lower)
- Increase `arxiv.max_results` to download more papers per search
- Modify `text_processing.chunk_size` for larger or smaller text chunks
- Change `storage.papers_directory` to store PDFs in a different location

### Environment Variables (Recommended for API Keys)

Store sensitive API keys in a `.env` file instead of `config.yaml`:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=sk-...
KIMI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...

# LMStudio (if needed)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

Environment variables take precedence over `config.yaml` settings and are more secure (`.env` is in `.gitignore`).

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

3. **Configure AI Provider**:
   - Edit `config.yaml` to select your provider (lmstudio, openai, kimi, or deepseek)
   - Add API keys to `.env` file if using cloud providers
   - If using LMStudio, ensure it's running with local server started

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

### Provider Connection Issues

**For LMStudio:**
1. Check that LMStudio is running
2. Verify the local server is started in LMStudio
3. Ensure the server is accessible at `http://localhost:1234`
4. Check that your models are loaded in LMStudio

**For Cloud Providers (OpenAI, Kimi, DeepSeek):**
1. Verify your API key is correct in `.env` or `config.yaml`
2. Check your account has credits/quota available
3. Ensure you're not hitting rate limits
4. Verify the model names are correct for your provider

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

### Version 3.0 (Latest)
- 🌐 **Multi-Provider Support**: Use OpenAI, Kimi, DeepSeek, or LMStudio
- 🔀 **Mix & Match**: Different providers for embeddings and chat
- 🔐 **Better Security**: API keys via environment variables
- ⚙️ **Flexible Config**: Easy provider switching in config.yaml
- 📚 **Provider Documentation**: Setup guides for all providers

### Version 2.0
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
