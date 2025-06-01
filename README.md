# Cybersecurity Awareness Assistant

A **Retrieval-Augmented Generation (RAG)** powered cybersecurity knowledge assistant that provides intelligent answers to security questions by leveraging uploaded PDF documents. Built with FastAPI, this system combines document retrieval with AI generation to deliver accurate, context-aware cybersecurity guidance.

##  Key Features

- **AI-Powered Q&A**: Ask cybersecurity questions and get intelligent, contextual answers
- **PDF Knowledge Base**: Upload cybersecurity documents to expand the system's knowledge
- **Smart Retrieval**: RAG architecture finds the most relevant information from your documents
-  **Fast API**: RESTful API built with FastAPI for high performance
- **Health Monitoring**: Built-in health checks and status endpoints
- **Topic Organization**: Predefined cybersecurity categories for better content management
-  **Testing Suite**: Automated tests to verify system functionality
-  **Extensible**: Easy to integrate with different vector stores and LLMs

## Use Cases

- **Security Training**: Interactive cybersecurity education platform
- **Help Desk Support**: Automated first-line security question answering
- **Policy Guidance**: Quick access to organizational security policies
- **Incident Response**: Rapid retrieval of security procedures and best practices
- **Compliance**: Easy lookup of regulatory requirements and standards

##  Architecture

This system implements a **Retrieval-Augmented Generation (RAG)** architecture:

```
User Query → Document Retrieval → Context Enrichment → AI Generation → Response
```

1. **Document Ingestion**: PDFs are processed and stored in a searchable format
2. **Query Processing**: User questions are analyzed and relevant context is retrieved
3. **Answer Generation**: AI model generates responses based on retrieved context
4. **Response Delivery**: Structured JSON response with sources and metadata

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended for optimal performance

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/GraceIrungu96/RAG-CyberSecurityQ-A-System](https://github.com/GraceIrungu96/RAG-CyberSecurityQ-A-System).git
   cd RAG-CyberSecurityQ-A-System
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The server will start at `http://localhost:8000` with automatic API documentation at `http://localhost:8000/docs`.

##  Dependencies

Create a `requirements.txt` file with the following packages:

```txt
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
pydantic>=1.8.0
python-multipart>=0.0.5
aiofiles>=0.7.0
PyPDF2>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## API Endpoints

### Query Endpoint
**POST** `/query`

Ask cybersecurity questions and receive AI-generated answers.

**Request:**
```json
{
  "question": "How can I identify a phishing email?",
  "context_limit": 3
}
```

**Response:**
```json
{
  "answer": "Phishing emails typically exhibit several warning signs: suspicious sender addresses, urgent language requesting immediate action, unexpected attachments or links, and requests for sensitive information...",
  "sources": [
    {
      "title": "Email Security Best Practices",
      "page": 5,
      "relevance_score": 0.95
    }
  ],
  "context_used": 3,
  "confidence": 0.87,
  "timestamp": "2025-06-01T12:00:00Z"
}
```

### Upload Endpoint
**POST** `/upload-pdf`

Upload PDF documents to expand the knowledge base.

**Form Data:**
- `file`: PDF file (required)
- `source_name`: Document identifier (required)
- `topic`: Category (optional)
- `difficulty`: beginner|intermediate|advanced (optional)

**Response:**
```json
{
  "message": "PDF uploaded and processed successfully",
  "document_id": "doc_12345",
  "pages_processed": 25,
  "chunks_created": 47
}
```
###  Topics Endpoint
**GET** `/topics`

Get available cybersecurity topic categories.

**Response:**
```json
{
  "topics": [
    "phishing",
    "passwords",
    "malware",
    "social_engineering",
    "network_security",
    "data_protection",
    "incident_response",
    "security_awareness",
    "compliance",
    "threat_intelligence"
  ]
}
```

###  Health Check
**GET** `/health`

Monitor system status and readiness.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-01T12:30:00Z",
  "rag_system_initialized": true,
  "documents_loaded": 15,
  "memory_usage": "2.3GB",
  "uptime": "2h 15m"
}
```

### ℹ API Information
**GET** `/`

Get API version and endpoint information.

## Testing

### Automated Testing
Run the built-in test suite:
```bash
python -m pytest tests/
```

### Manual Testing
The application includes a `test_system()` function that runs automatically on startup:

```python
def test_system():
    """Test basic RAG functionality"""
    test_questions = [
        "What is social engineering?",
        "How do I create a strong password?",
        "What are signs of malware infection?"
    ]
    # Tests run automatically...
```

### API Testing with curl

```bash
# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is two-factor authentication?"}'

# Upload a PDF
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@security_guide.pdf" \
  -F "source_name=Security Guidelines" \
  -F "topic=general"
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# RAG Configuration
MAX_CONTEXT_LENGTH=4000
SIMILARITY_THRESHOLD=0.7
MAX_DOCUMENTS_PER_QUERY=5

# Upload Configuration
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### Customizing the RAG System

Modify the `CybersecurityRAGSystem` class to integrate with different backends:

```python
class CybersecurityRAGSystem:
    def __init__(self):
        # Initialize your preferred vector store
        # Options: FAISS, Pinecone, Chroma, Weaviate
        pass
    
    def query(self, question: str, context_limit: int = 3):
        # Implement your RAG logic
        pass
    
    def add_pdf_document(self, file_path: str, source_name: str, metadata: dict):
        # Document processing and indexing
        pass
```

## Deployment

### Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and run:**
```bash
docker build -t cybersec-rag-assistant .
docker run -p 8000:8000 cybersec-rag-assistant
```

### Cloud Deployment Options

- **Heroku**: Easy deployment with git push
- **AWS Lambda**: Serverless deployment with API Gateway
- **Google Cloud Run**: Containerized serverless deployment
- **DigitalOcean App Platform**: Simple container deployment
- **Railway**: Git-based deployment platform

##  Security Considerations

**Important Security Notes:**

- This tool is for **educational and informational purposes only**
- Do not rely solely on AI-generated responses for critical security decisions
- Always verify information with authoritative sources
- Implement proper authentication and authorization in production
- Use HTTPS in production environments
- Regularly update dependencies to patch security vulnerabilities
- Sanitize user inputs to prevent injection attacks
- Implement rate limiting to prevent abuse

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 Python style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for API changes
- Use meaningful commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for RAG research and methodologies
- **LangChain** for inspiration on document processing
- **FastAPI** community for excellent documentation
- **Cybersecurity professionals** who contribute to open knowledge sharing

## Additional Resources

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SANS Security Awareness](https://www.sans.org/security-awareness-training/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Support & Contact

**Developer:** Grace W. Irungu  
**Email:** graceirungu96@gmail.com  
**LinkedIn:** [Connect with Grace](https://linkedin.com/in/graceirungu)

For bug reports and feature requests, please use the [GitHub Issues](https://github.com/yourusername/cybersecurity-rag-assistant/issues) page.

---
