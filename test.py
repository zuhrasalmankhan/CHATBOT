import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from langchain.schema import Document
from RAGfile import RAGTool  # Replace 'your_module' with the actual module name

class TestRAGTool(unittest.TestCase):

    @patch('RAGfile.MongoClient')
    @patch('RAGfile.TogetherEmbeddings')
    def setUp(self, mock_embeddings, mock_mongo):
        # Mock environment variables
        with patch.dict('os.environ', {
            'TOGETHER_API_KEY': 'fake_api_key',
            'MONGOURI': 'fake_mongouri'
        }):
            # Mock MongoDB collection
            self.mock_collection = Mock()
            mock_mongo.return_value.__getitem__.return_value.__getitem__.return_value = self.mock_collection
            
            # Initialize RAGTool
            self.rag_tool = RAGTool("dummy_path.pdf")
            
            # Mock embeddings
            self.mock_embeddings = mock_embeddings.return_value
            self.rag_tool.embeddings = self.mock_embeddings

    @patch('RAGfile.PDFPlumberLoader')
    @patch('RAGfile.RecursiveCharacterTextSplitter')
    def test_store_embeddings_when_collection_empty(self, mock_splitter, mock_loader):
        # Mock empty collection
        self.mock_collection.count_documents.return_value = 0
        
        # Mock PDF loader
        mock_docs = [Document(page_content="Sample text")]
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance
        
        # Mock text splitter
        mock_chunks = [Document(page_content="Sample chunk")]
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        # Mock embeddings generation
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Call the method
        self.rag_tool.store_embeddings_once()
        
        # Assertions
        self.mock_collection.insert_one.assert_called_once_with({
            "text": "Sample chunk",
            "embedding": [0.1, 0.2, 0.3]
        })

    def test_store_embeddings_when_collection_not_empty(self):
        # Mock non-empty collection
        self.mock_collection.count_documents.return_value = 1
        
        # Call the method
        self.rag_tool.store_embeddings_once()
        
        # Assertions
        self.mock_collection.insert_one.assert_not_called()

    def test_retrieve_documents(self):
        # Mock query embedding
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock stored documents with embeddings
        mock_docs = [
            {"text": "Doc 1", "embedding": [0.4, 0.5, 0.6]},
            {"text": "Doc 2", "embedding": [0.1, 0.2, 0.3]}  # Exact match for query
        ]
        self.mock_collection.find.return_value = mock_docs
        
        # Call retrieve method
        results = self.rag_tool.retrieve("test query", top_k=1)
        
        # Assertions
        self.assertEqual(results, ["Doc 2"])  # Most similar document
        self.mock_embeddings.embed_query.assert_called_once_with("test query")

    def test_retrieve_with_empty_collection(self):
        self.mock_collection.find.return_value = []
        results = self.rag_tool.retrieve("test query")
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()