from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv

import os

load_dotenv()
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

MONGOURI = os.environ.get("MONGOURI")
client = MongoClient(MONGOURI)
db = client["your_database_name"] # Replace 'your_database_name' with the actual name of your MongoDB database
collection = db['collection_name']
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")



loader = PDFPlumberLoader("netsol_report.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,    
    )

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[100].page_content}")

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
    )
print ("Embedding done")

for chunk, vector in zip(chunks, vectors):
    doc = {
        "text": chunk.page_content,
        "embedding": vector  # List of floats
    }
    collection.insert_one(doc)

print("Embeddings stored in MongoDB Atlas.")

