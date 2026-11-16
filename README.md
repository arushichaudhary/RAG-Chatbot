# University RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about ABC University using its own knowledge base instead of relying only on the LLM's memory.

## How It Works

1. The university document (`data/university.txt`) is loaded and split into smaller chunks.
2. Each chunk is converted into a numerical embedding using a Sentence Transformers model.
3. The embeddings are stored in a FAISS vector database.
4. When a user asks a question, it is converted into an embedding and compared against stored chunks to retrieve the three most relevant results.
5. The retrieved information is inserted into a prompt and sent to Gemini to generate a context-based answer.

## Project Structure

```text
RAG_Chatbot/
├── data/
│   └── university.txt
├── evaluation/
│   ├── test_questions.json
│   └── evaluate.py
├── vector_db/
├── ingest.py
├── rag_chatbot.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key (Optional)

Set your Gemini API key to generate answers using the LLM.

Get an API key from:
https://aistudio.google.com/apikey

Windows:
```bash
set GEMINI_API_KEY=your_key_here
```

Linux/Mac:
```bash
export GEMINI_API_KEY=your_key_here
```

## Usage

### 1. Build the Vector Database

Run this once, or whenever you update `data/university.txt`.

```bash
python ingest.py
```

### 2. Chat from the Terminal

```bash
python rag_chatbot.py
```

### 3. Launch the Web Interface

```bash
streamlit run app.py
```

## Answer Quality Evaluation

In addition to retrieval evaluation, generated answers are manually compared against reference answers from the university knowledge base.

The following aspects are checked:

| Evaluation Criteria | Description |
|---|---|
| Factual correctness | Whether the answer matches the source document |
| Context relevance | Whether the answer addresses the question |
| Completeness | Whether important information is included |
| Groundedness | Whether the answer is supported by retrieved context |
| Unsupported questions | Whether the chatbot avoids inventing information |

Answers are marked as Correct, Partially Correct, or Incorrect after comparison with the reference information.

Out-of-scope questions are checked separately to evaluate whether the chatbot follows its missing-evidence instructions.

## Chunking Strategy

The project uses LangChain's RecursiveCharacterTextSplitter to divide the university document into smaller chunks.

Current configuration:

- Chunk size: 500 characters
- Chunk overlap: 50 characters

A 500-character chunk keeps the retrieved information relatively focused, while a 50-character overlap helps preserve context between consecutive chunks.

These values were selected as initial configuration choices and can be adjusted based on retrieval evaluation.

## Retrieval and Ranking Strategy

The project uses FAISS for similarity-based retrieval.

1. The user query is converted into an embedding using the same Sentence Transformers model used during ingestion.
2. FAISS compares the query embedding with stored document embeddings.
3. The three most similar chunks are retrieved.
4. Retrieved chunks are combined into a context.
5. The context and user question are passed to Gemini 2.5 Flash.

The current implementation uses Top-3 similarity retrieval to balance relevant context and prompt size.

FAISS distance scores are also inspected during evaluation to understand retrieval behaviour.

## Sample Inputs and Expected Outputs

The following examples are based on the information available in `data/university.txt`. Outputs are illustrative and have not yet been verified through a live execution.

### Example 1: Hostel Information

**Input:**
```text
You: What is the hostel fee?
```

**Expected Output:**
```text
Bot: The hostel fee is Rs. 20,000 per semester.
```

### Example 2: Admission Information

**Input:**
```text
You: When does admission open?
```

**Expected Output:**
```text
Bot: Admission at ABC University opens on July 15.
```

### Example 3: Academic Information

**Input:**
```text
You: What courses does the Computer Science department offer?
```

**Expected Output:**
```text
Bot: The Computer Science department offers
Machine Learning, Data Structures, Operating Systems,
and Web Development.
```

### Example 4: Question Outside the Knowledge Base

**Input:**
```text
You: Who won the FIFA World Cup in 2022?
```

**Expected Output (with Gemini API configured):**
```text
Bot: I don't know.
```

The prompt instructs the LLM to answer only using the supplied context and respond with "I don't know" when the information is unavailable.

**Note:** These are expected responses based on the knowledge-base content and prompt instructions, not outputs from a completed test. The current implementation does not include a separate similarity threshold to guarantee rejection of unrelated questions.

## My Contributions

I independently designed and developed this RAG chatbot, implementing its document processing, retrieval pipeline, LLM integration, and user interface.

My contributions include:

- Created the university knowledge-base document containing information about admissions, fees, hostels, academics, examinations, and placements.
- Implemented document loading and text chunking using LangChain's `TextLoader` and `RecursiveCharacterTextSplitter`.
- Integrated Sentence Transformers to generate semantic embeddings using the `all-MiniLM-L6-v2` model.
- Implemented FAISS vector storage and similarity-based retrieval to find relevant information for user queries.
- Developed the RAG pipeline to combine retrieved document chunks with user questions and construct context-based prompts.
- Integrated Google's Gemini 2.5 Flash API for natural-language answer generation.
- Developed a Streamlit-based chatbot interface with chat history, custom chat bubbles, and an option to view retrieved document chunks.
- Implemented an API-key fallback so that the application can display retrieved context when a Gemini API key is not configured.

## Technical Decisions

- **Sentence Transformers (`all-MiniLM-L6-v2`):** Selected to generate semantic embeddings that help identify relevant information based on the meaning of a query.
- **FAISS:** Used for efficient similarity-based retrieval of relevant document chunks.
- **Chunking Strategy:** Used a chunk size of 500 characters with an overlap of 50 characters to retain some context between consecutive chunks.
- **Top-K Retrieval:** Configured the retriever to return the three most relevant chunks for each query.
- **RAG Architecture:** Used retrieval-augmented generation to provide the LLM with relevant university information before generating an answer.
- **Gemini 2.5 Flash:** Integrated for generating natural-language responses using the retrieved context.
- **Streamlit:** Chosen to build an interactive web interface without requiring a separate frontend application.

## Notes

- If `GEMINI_API_KEY` is not set, the chatbot displays the retrieved context and prompt instead of a generated answer.
- Additional information can be added to `data/university.txt`, followed by rebuilding the vector database using `ingest.py`.
- The web interface uses a pastel color theme with a soft lavender background, pink user bubbles, and mint bot bubbles.
- The application also displays retrieved document chunks to make the retrieval process easier to inspect.

## Deploying on Render

1. Push the project to a GitHub repository. The `vector_db/` and `venv/` directories are excluded using `.gitignore`.
2. On Render, create a new Web Service and connect your GitHub repository.
3. Configure the following:

**Build Command:**
```bash
pip install -r requirements.txt && python ingest.py
```

**Start Command:**

Use the command configured in the `Procfile` to start the Streamlit application.

4. Under Environment Variables, add:

```text
GEMINI_API_KEY = your_gemini_key
```

5. Deploy the application. Render will build the vector database and start the Streamlit application.
6. Once deployment is complete, Render provides a public URL to access the chatbot.
