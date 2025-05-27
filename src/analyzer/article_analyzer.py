from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class ArticleAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the article")
    topics: List[str] = Field(description="List of main topics discussed in the article")
    key_points: List[str] = Field(description="List of key points from the article")

class ArticleAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=ArticleAnalysis)
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert news analyst. Analyze the following article and provide:
            1. A concise summary (2-3 sentences)
            2. Main topics discussed
            3. Key points
            
            {format_instructions}"""),
            ("user", "Article Title: {title}\n\nArticle Text: {text}")
        ])

    def analyze_article(self, title: str, text: str) -> ArticleAnalysis:
        """
        Analyze an article using GPT to generate summary, topics, and key points.
        
        Args:
            title (str): The article title
            text (str): The article text
            
        Returns:
            ArticleAnalysis: Object containing the analysis results
        """
        try:
            # Format the prompt with the article content
            prompt = self.analysis_prompt.format_messages(
                title=title,
                text=text,
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Get the analysis from GPT
            response = self.llm.invoke(prompt)
            
            # Parse the response into our Pydantic model
            analysis = self.parser.parse(response.content)
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Failed to analyze article: {str(e)}") 