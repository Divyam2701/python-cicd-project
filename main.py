# import io
# import os
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader, errors
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import FakeEmbeddings
# from langchain_community.vectorstores import Qdrant
# from langchain.chains import RetrievalQA
# from langchain.llms.fake import FakeListLLM
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS (for local frontend testing)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory Qdrant setup
# client = QdrantClient(":memory:")
# collection_name = "pdf_collection"

# client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
# )

# # Upload and process PDF
# @app.post("/upload/")
# async def upload(file: UploadFile = File(...)):
#     try:
#         content = await file.read()
#         reader = PdfReader(io.BytesIO(content))
#         text = "".join([page.extract_text() or "" for page in reader.pages])

#         if not text.strip():
#             raise HTTPException(status_code=400, detail="PDF has no extractable text.")

#         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         texts = splitter.split_text(text)

#         # Use fake embeddings for testing instead of OpenAI
#         embeddings = FakeEmbeddings(size=1536)
#         Qdrant.from_texts(texts, embeddings, client=client, collection_name=collection_name)

#         return {"status": "uploaded and processed"}

#     except errors.PdfReadError as e:
#         raise HTTPException(status_code=400, detail=f"PDF Read Error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

# # Question model
# class Query(BaseModel):
#     question: str

# # Answer questions
# @app.post("/ask/")
# async def ask(query: Query):
#     try:
#         retriever = Qdrant(client=client, collection_name=collection_name).as_retriever()
#         chain = RetrievalQA.from_chain_type(
#             llm=FakeListLLM(responses=["This is a fake answer."]),
#             chain_type="stuff",
#             retriever=retriever
#         )
#         answer = chain.run(query.question)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# # Lambda-compatible handler
# def lambda_handler(event, context):
#     from mangum import Mangum
#     handler = Mangum(app)
#     return handler(event, context)


import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader, errors
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms.fake import FakeListLLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- Setup ---
app = FastAPI()

# CORS (adjust if deployed publicly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory vector store
client = QdrantClient(":memory:")
collection_name = "pdf_collection"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# --- API Routes ---
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = "".join([page.extract_text() or "" for page in reader.pages])

        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found.")

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_text(text)

        embeddings = FakeEmbeddings(size=1536)
        Qdrant.from_texts(texts, embeddings, client=client, collection_name=collection_name)

        return {"status": "PDF processed successfully"}

    except errors.PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"PDF Read Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask(query: Query):
    try:
        retriever = Qdrant(client=client, collection_name=collection_name).as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=FakeListLLM(responses=["This is a fake answer."]),
            chain_type="stuff",
            retriever=retriever
        )
        answer = chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Lambda Adapter ---
from mangum import Mangum
lambda_handler = Mangum(app)
