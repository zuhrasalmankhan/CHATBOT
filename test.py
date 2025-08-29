import os
from RAGfile import RAGTool
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def test_mongo_connection():
    """Test MongoDB Atlas connection before doing anything else."""
    try:
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("MONGO_URI not found in environment variables.")

        # Connect to Atlas cluster
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a server selection to verify connection
        dbs = client.list_database_names()
        print("✅ MongoDB Atlas connected successfully!")
        print("Databases available:", dbs)
        return True
    except Exception as e:
        print("❌ MongoDB Atlas connection failed:", str(e))
        return False

def main():
    # Test MongoDB Atlas connection before proceeding
    if not test_mongo_connection():
        print("Exiting program due to MongoDB connection error.")
        return

    # Path to your PDF
    pdf_path = "Netsol_report.pdf"
    
    # Initialize RAG tool with Atlas connection
    MONGO_URI = os.getenv("MONGO_URI")
    rag_tool = RAGTool(pdf_path=pdf_path)

    # Example query
    query = "What does the report say about AI regulations?"
    results = rag_tool.retrieve(query)

    if results:
        print("\nTop results:")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(f"Text: {r['text'][:300]}...\n")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
