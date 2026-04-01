# 🩺 MediGuide AI - Medical Chatbot Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.26-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**MediGuide AI** is an intelligent medical chatbot assistant powered by advanced LLM technology. It provides reliable, evidence-based answers from trusted medical encyclopedias covering diseases, treatments, mental health, nutrition, pediatrics, and general wellness topics.

Built by **Riflan Mohamed** | 2025

---

## 🌟 Features

- **🤖 AI-Powered Responses** - Uses OpenAI GPT-4o for natural, contextual medical guidance
- **📚 Medical Knowledge Base** - Retrieval-Augmented Generation (RAG) from medical encyclopedias stored in Pinecone vector database
- **💬 Conversational Memory** - Remembers chat context for follow-up questions within each session
- **🎨 Modern UI** - Beautiful, responsive interface with dark/light themes
- **⚡ Real-time Responses** - Fast semantic search and answer generation
- **📱 Fully Responsive** - Works seamlessly on desktop, tablet, and mobile devices
- **🔒 Session Management** - Isolated conversation memory per user

---

## 🏗️ Architecture

```
User Query → Flask App → LangChain ConversationalRetrievalChain
                ↓
         Pinecone Vector DB (Medical PDFs embedded)
                ↓
         Retrieve relevant medical context (top 3 chunks)
                ↓
         OpenAI GPT-4o generates answer using context + chat history
                ↓
         Response returned to user
```

**Tech Stack:**
- **Backend:** Flask (Python)
- **LLM:** OpenAI GPT-4o
- **Vector Database:** Pinecone (Serverless)
- **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Framework:** LangChain for orchestration
- **Frontend:** HTML, CSS (custom design), JavaScript

---

## 📋 Prerequisites

Before running this project, you need:

### 1. **Python 3.11+**
Download from [python.org](https://www.python.org/downloads/)

### 2. **Pinecone Account** (Free tier available)
- Sign up at [pinecone.io](https://www.pinecone.io/)
- Create a new project
- Get your **API Key** from the dashboard
- Note: This project uses **Serverless deployment** on AWS `us-east-1`

### 3. **OpenAI API Key**
- Sign up at [platform.openai.com](https://platform.openai.com/)
- Create an API key from your account settings
- Billing must be set up (GPT-4o requires paid credits)

### 4. **Medical PDF Documents** (Optional for custom data)
- Place PDF files in a `data/` folder at the root
- Supported: Medical encyclopedias, research papers, health guides
- The more comprehensive your PDFs, the better the chatbot's knowledge

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Medi-Guide-Chat-Bot.git
cd Medi-Guide-Chat-Bot
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `flask==3.1.1` - Web framework
- `langchain==0.3.26` - LLM orchestration
- `langchain-openai==0.3.24` - OpenAI integration
- `langchain-pinecone==0.2.8` - Pinecone vector store
- `langchain-huggingface==0.1.2` - HuggingFace embeddings
- `sentence-transformers==4.1.0` - Embedding model
- `pypdf==5.6.1` - PDF parsing
- `python-dotenv==1.1.0` - Environment variables
- `gunicorn==20.1.0` - Production server

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Port configuration (default: 8080)
PORT=8080
```

⚠️ **Never commit your `.env` file to Git!** It's already in `.gitignore`.

---

## 📂 Preparing Medical Data

### Step 1: Add Your PDF Files

Create a `data/` folder and add medical PDFs:

```bash
mkdir data
# Copy your medical encyclopedia PDFs into this folder
# Examples: Gale Encyclopedia of Medicine, nursing guides, etc.
```

### Step 2: Create Pinecone Index & Upload Data

**Run this script ONCE locally** (not on deployment):

```bash
python store_index.py
```

**What this does:**
1. Loads all PDFs from the `data/` folder
2. Splits documents into 500-character chunks (20-char overlap)
3. Creates embeddings using `all-MiniLM-L6-v2` (384 dimensions)
4. Creates a Pinecone index named `medi-guide-bot`
5. Uploads vectors in batches of 80 documents

**Pinecone Index Configuration:**
- **Name:** `medi-guide-bot`
- **Dimension:** 384 (matches embedding model)
- **Metric:** Cosine similarity
- **Cloud:** AWS Serverless
- **Region:** us-east-1

⏱️ **Note:** For large PDFs, this process can take 10-30 minutes. You only need to run this once.

---

## 🖥️ Running the Application

### Local Development

```bash
python app.py
```

The app will start on `http://localhost:8080`

### Production Mode (Gunicorn)

```bash
gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --timeout 120
```

---

## 🌐 Deployment

### Deploy to Render (Recommended)

1. **Push to GitHub** (if not already)
2. **Create a new Web Service** on [render.com](https://render.com)
3. **Connect your GitHub repository**
4. **Configure build settings:**
   - **Build Command:** `pip install --no-cache-dir -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. **Add environment variables:**
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
   - `PYTHON_VERSION=3.11.9`
6. **Deploy!**

Render configuration is already provided in `render.yaml`.

### Deploy to Vercel (Alternative)

**Note:** The `vercel.json` config exists but requires serverless function setup in `api/index.py`. Not recommended for this Flask app due to cold start issues with embeddings.

---

## 📁 Project Structure

```
Medi-Guide-Chat-Bot/
│
├── app.py                  # Main Flask application (backend)
├── store_index.py          # Script to create Pinecone index & upload data
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── render.yaml             # Render deployment configuration
├── setup.py                # Package setup file
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── helper.py           # PDF loading, text splitting, embeddings
│   └── prompt.py           # System prompt for the AI
│
├── templates/              # HTML templates
│   └── chat.html           # Main chat interface
│
├── static/                 # Static assets
│   ├── styles.css          # Custom CSS (dark/light themes)
│   └── mg.png              # Logo/favicon
│
├── data/                   # Medical PDF documents (not in Git)
│   └── *.pdf               # Add your medical encyclopedias here
│
├── .env                    # Environment variables (not in Git)
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🔧 How It Works

### 1. **Document Loading** (`src/helper.py`)
- `load_pdf_file()`: Loads all PDFs from the `data/` directory
- Uses `PyPDFLoader` to extract text from each page

### 2. **Text Chunking** (`src/helper.py`)
- `text_split()`: Splits documents into smaller chunks
- **Chunk size:** 500 characters
- **Overlap:** 20 characters (ensures context continuity)

### 3. **Embedding Generation** (`src/helper.py`)
- `download_hugging_face_embeddings()`: Loads embedding model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Runs on CPU (optimized for Render free tier)

### 4. **Vector Storage** (`store_index.py`)
- Embeds all text chunks into 384-dimensional vectors
- Stores in Pinecone index `medi-guide-bot`
- Enables semantic similarity search

### 5. **Conversational Retrieval** (`app.py`)
- User sends a question → Flask `/get` endpoint
- **Retriever** searches Pinecone for top 3 most similar chunks
- **Memory** stores chat history per session
- **LangChain ConversationalRetrievalChain** combines:
  - Retrieved medical context
  - Conversation history
  - System prompt
- **OpenAI GPT-4o** generates the final answer
- Response is sent back to the user

### 6. **Frontend** (`templates/chat.html`, `static/styles.css`)
- Modern chat interface with message bubbles
- Dark/light theme toggle
- Suggestions and knowledge categories
- Typing indicators
- Fully responsive design

---

## 🔑 Environment Variables Explained

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | ✅ Yes | Your Pinecone API key from [pinecone.io](https://pinecone.io) |
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key from [platform.openai.com](https://platform.openai.com) |
| `PORT` | ❌ No | Server port (default: 8080). Set automatically by hosting platforms. |

---

## 🧪 Testing the Chatbot

### Sample Questions to Try:

- "What are the early signs of diabetes?"
- "How can I treat anxiety naturally?"
- "What's the recommended vaccination schedule for children?"
- "What foods are good for heart health?"
- "How do I lower blood pressure without medication?"
- "What are common symptoms of the flu?"

The chatbot will retrieve relevant information from your medical PDFs and generate evidence-based answers.

---

## 🛠️ Customization

### Change Embedding Model

Edit `src/helper.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name='your-preferred-model',  # e.g., 'sentence-transformers/all-mpnet-base-v2'
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

⚠️ **Important:** If you change the model, you must update the Pinecone index dimension to match.

### Change LLM Model

Edit `app.py`:

```python
chat_model = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-3.5-turbo" for lower cost
    temperature=0.3,
    max_retries=2,
    request_timeout=60
)
```

### Adjust Retrieval Settings

Edit `app.py`:

```python
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 chunks instead of 3
)
```

### Modify System Prompt

Edit `src/prompt.py`:

```python
system_prompt = (
    "You are MediGuide, an AI-powered medical assistant. "
    "Your custom instructions here..."
)
```

---

## 🐛 Troubleshooting

### Issue: `PINECONE_API_KEY must be set`
**Solution:** Create a `.env` file with valid API keys.

### Issue: `Error connecting to Pinecone`
**Solution:** 
1. Verify your API key is correct
2. Ensure you've run `store_index.py` to create the index
3. Check if index name is `medi-guide-bot`

### Issue: `Empty message`
**Solution:** The chatbot requires non-empty input. Check your frontend is sending the `msg` parameter correctly.

### Issue: Slow responses
**Solution:**
- Consider using `gpt-3.5-turbo` instead of `gpt-4o`
- Reduce `k` value in retriever (fetch fewer chunks)
- Upgrade to a paid Render plan for more resources

### Issue: Out of memory errors
**Solution:**
- Reduce batch size in `store_index.py` (default: 80)
- Use smaller PDF files
- Upgrade server resources

---

## 📊 Pinecone Index Details

### Current Configuration:
- **Index Name:** `medi-guide-bot`
- **Dimension:** 384 (matches `all-MiniLM-L6-v2`)
- **Metric:** Cosine similarity
- **Cloud Provider:** AWS
- **Region:** us-east-1
- **Deployment:** Serverless (free tier)

### Pinecone Free Tier Limits:
- **Vectors:** 100,000 vectors
- **Storage:** Limited by vector count
- **Queries:** Unlimited

For larger medical knowledge bases, consider upgrading to a paid plan.

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Built by Riflan Mohamed**  
📧 Email: rizlanahmd4545@gmail.com  
🔗 GitHub: [Your GitHub Profile]

© 2025 MediGuide AI. All rights reserved.

---

## 🙏 Acknowledgments

- **OpenAI** - GPT-4o language model
- **Pinecone** - Vector database infrastructure
- **HuggingFace** - Embedding models
- **LangChain** - LLM orchestration framework
- **Flask** - Web framework

---

## ⚠️ Disclaimer

**MediGuide AI is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider for medical concerns.**

---

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] PDF export of conversations
- [ ] Integration with health APIs
- [ ] User authentication
- [ ] Chat history persistence
- [ ] Admin dashboard for monitoring

---

**Star ⭐ this repository if you find it useful!**
