import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import json
import os

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory (str): Directory to persist the database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"}
        )

    def add_article(self, 
                   article_id: str,
                   title: str,
                   text: str,
                   summary: str,
                   topics: List[str],
                   metadata: Dict[str, Any]) -> None:
        """
        Add an article to the vector store.
        
        Args:
            article_id (str): Unique identifier for the article
            title (str): Article title
            text (str): Full article text
            summary (str): Article summary
            topics (List[str]): List of topics
            metadata (Dict[str, Any]): Additional metadata
        """
        # Combine text for embedding
        combined_text = f"{title}\n\n{summary}\n\n{' '.join(topics)}"
        
        # Add to collection
        self.collection.add(
            ids=[article_id],
            documents=[combined_text],
            metadatas=[{
                **metadata,
                "title": title,
                "summary": summary,
                "topics": json.dumps(topics)
            }]
        )

    def search_articles(self, 
                       query: str,
                       n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles using semantic similarity.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching articles with their metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        articles = []
        for i in range(len(results['ids'][0])):
            article = {
                'id': results['ids'][0][i],
                'title': results['metadatas'][0][i]['title'],
                'summary': results['metadatas'][0][i]['summary'],
                'topics': json.loads(results['metadatas'][0][i]['topics']),
                'similarity': results['distances'][0][i] if 'distances' in results else None
            }
            articles.append(article)
            
        return articles 