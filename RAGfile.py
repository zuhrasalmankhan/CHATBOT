from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from pymongo import MongoClient
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MONGOURI = os.environ.get("MONGOURI")

class RAGTool:
    def __init__(self, pdf_path: str, db_name: str = "your_database_name", collection_name: str = "collection_name"):
        self.pdf_path = pdf_path
        self.client = MongoClient(MONGOURI)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")
    
    def store_embeddings_once(self, chunk_size=1000, chunk_overlap=200):
        if self.collection.count_documents({}) > 0:
            print("Embeddings already exist in MongoDB. Skipping embedding.")
            return

        loader = PDFPlumberLoader(self.pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_documents(documents)

        vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])

        for chunk, vector in zip(chunks, vectors):
            doc = {
                "text": chunk.page_content,
                "embedding": vector
            }
            self.collection.insert_one(doc)

        print(f"Stored {len(chunks)} embeddings in MongoDB.")

    def retrieve(self, query: str, top_k: int = 3):
        # Embed the query
        query_vector = np.array(self.embeddings.embed_query(query))

        # Fetch stored embeddings
        docs = list(self.collection.find({}, {"text": 1, "embedding": 1, "_id": 0}))

        # Compute cosine similarity
        results = []
        for doc in docs:
            doc_vector = np.array(doc["embedding"])
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            results.append((similarity, doc["text"]))

        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in results[:top_k]]
