# Document Q&A Assistant

A production-ready document-based question-answering application built with Amazon Bedrock, LangChain, and Streamlit. This application uses local vector storage (FAISS) for efficient document retrieval and conversational AI capabilities.

## Features

- **Document Upload**: Support for PDF document upload via Streamlit interface
- **Local Vector Storage**: FAISS-based vector store for fast similarity search
- **Conversational Interface**: Chat-based Q&A with message history
- **Flexible Configuration**: Environment-based configuration for easy deployment
- **Source Document Display**: View retrieved document segments with relevance scores
- **Multiple Bedrock Models**: Support for various Amazon Bedrock foundation models

## Prerequisites

- Python 3.9+
- AWS account with Bedrock access
- AWS credentials configured (via AWS CLI or environment variables)
- PDF documents for Q&A

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ernesthenry/document-qa-assistant
cd learning-amazon-bedrock-3819146
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set your configuration values:
- `MODEL_ID` (default: anthropic.claude-instant-v1)
- `DEFAULT_PDF_PATH` (path to default PDF document)
- Model parameters (temperature, max_tokens, etc.)

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `PAGE_TITLE` | Browser page title | Document Q&A Assistant | - |
| `APP_TITLE` | Application title | Document Q&A Assistant | - |
| `MODEL_ID` | Bedrock model ID | anthropic.claude-instant-v1 | - |
| `MAX_TOKENS` | Maximum tokens to generate | 512 | - |
| `TEMPERATURE` | Model temperature | 0.1 | 0.0-1.0 |
| `TOP_P` | Top P sampling | 1.0 | - |
| `DEFAULT_PDF_PATH` | Default PDF document path | documents/sample.pdf | - |
| `TOP_K_RESULTS` | Number of document results | 4 | - |

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Upload Document**: Use the sidebar to upload a PDF file or use the default document
2. **Ask Questions**: Type questions in the chat interface about the document content
3. **View Sources**: Enable "Show source documents" to see retrieved document segments
4. **Review History**: Conversation history is maintained during the session

## Project Structure

```
learning-amazon-bedrock-3819146/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
└── README.md          # This file
```

## How It Works

1. **Document Processing**: PDF documents are loaded and split into pages
2. **Embedding Generation**: Text segments are converted to vector embeddings using Bedrock
3. **Vector Storage**: Embeddings are stored in FAISS for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against document vectors
5. **Response Generation**: Retrieved context is provided to the LLM for answer generation

## License

See LICENSE file for details.
