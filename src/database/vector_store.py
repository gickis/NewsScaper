import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class VectorStore:
    """Vector store for article storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "data/chroma") -> None:
        """
        Initialize the vector store.
        
        Args:
            persist_directory (str): Directory to persist the database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM for query enhancement
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template for query enhancement
        self.query_enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding search queries and enhancing them for better semantic search results.
            Given a user query, generate a list of semantically related terms and concepts that would help find relevant articles.
            Focus on:
            1. Synonyms and related terms
            2. Broader and narrower concepts
            3. Context-specific terminology
            4. Domain-specific jargon and technical terms
            5. Common variations and alternative phrasings
            
            Return the enhanced query as a single string that combines the original query with the additional terms.
            Keep the enhanced query concise but comprehensive.
            Ensure the enhanced query maintains the original intent while expanding the search scope appropriately."""),
            ("user", "{query}")
        ])

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
        try:
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
        except Exception as e:
            raise Exception(f"Failed to add article to vector store: {str(e)}")

    def enhance_query(self, query: str) -> str:
        """
        Enhance the search query using LLM to include semantically related terms.
        
        Args:
            query (str): Original search query
            
        Returns:
            str: Enhanced query with additional semantic context
        """
        try:
            # Format the prompt with the query
            prompt = self.query_enhancement_prompt.format_messages(
                query=query
            )
            
            # Get enhanced query from LLM
            response = self.llm.invoke(prompt)
            enhanced_query = response.content.strip()
            
            return enhanced_query
        except Exception as e:
            # If enhancement fails, return original query
            return query

    def search_articles(self, 
                       query: str,
                       n_results: int = 5,
                       enhance_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search for articles using semantic similarity with enhanced query understanding.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            enhance_query (bool): Whether to enhance the query using LLM
            
        Returns:
            List[Dict[str, Any]]: List of matching articles with their metadata
        """
        try:
            # Enhance the query if requested
            search_query = self.enhance_query(query) if enhance_query else query
            
            # Perform the search
            results = self.collection.query(
                query_texts=[search_query],
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
                    'similarity': results['distances'][0][i] if 'distances' in results else None,
                    'original_query': query,
                    'enhanced_query': search_query if enhance_query else None
                }
                articles.append(article)
                
            return articles
        except Exception as e:
            raise Exception(f"Failed to search articles: {str(e)}") 