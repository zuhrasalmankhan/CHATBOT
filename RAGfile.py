from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv


load_dotenv()
MONGOURI = os.environ.get("MONGO_URI")


class RAGTool:
    def __init__(self, pdf_path: str, db_name: str = "CHATBOT", collection_name: str = "Chatbot"):
        self.pdf_path = pdf_path
        self.client = MongoClient(MONGOURI)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embeddings = SentenceTransformer("all-mpnet-base-v2")

    
    def store_embeddings_once(self, chunk_size=1000, chunk_overlap=200):
        """Embed the PDF and store in Mongo only if not already stored."""
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

        vectors = self.embeddings.encode([chunk.page_content for chunk in chunks])

        for chunk, vector in zip(chunks, vectors):
            doc = {
                "text": chunk.page_content,
                "embedding": vector.tolist()  # store as list for MongoDB
            }
            self.collection.insert_one(doc)

        print(f" Stored {len(chunks)} embeddings in MongoDB.")

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve most relevant text chunks for a query."""
        # Embed the query
        query_vector = np.array(self.embeddings.encode(query))

        # Fetch stored embeddings
        docs = list(self.collection.find({}, {"text": 1, "embedding": 1, "_id": 0}))
        if not docs:
            print("No documents found in MongoDB. Did you run store_embeddings_once()?")
            return []

        results = []
        for doc in docs:
            doc_vector = np.array(doc["embedding"])
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            results.append((similarity, doc["text"]))

        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)

        return [{"text": text, "score": float(score)} for score, text in results[:top_k]]
def main():
    pdf_path = "netsol_report.pdf"   
    rag = RAGTool(pdf_path)

    # Store embeddings once
    rag.store_embeddings_once()

    print("\n Ask questions about the PDF (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        answers = rag.retrieve(query)
        if not answers:
            print("No results found.")
        else:
            print("\nTop results:")
            for ans in answers:
                print(f"- (score: {ans['score']:.4f}) {ans['text'][:200]}...\n")  # preview first 200 chars


if __name__ == "__main__":
    main()
