from newspaper import Article
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleScraper:
    def __init__(self):
        self.article = None

    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        Scrape an article from the given URL.
        
        Args:
            url (str): The URL of the article to scrape
            
        Returns:
            Dict[str, str]: Dictionary containing article data
        """
        try:
            self.article = Article(url)
            self.article.download()
            self.article.parse()
            
            # Extract article data
            article_data = {
                'url': url,
                'title': self.article.title,
                'text': self.article.text,
                'publish_date': str(self.article.publish_date) if self.article.publish_date else None,
                'keywords': self.article.keywords,
                'summary': self.article.summary
            }
            
            logger.info(f"Successfully scraped article: {url}")
            return article_data
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {str(e)}")
            raise Exception(f"Failed to scrape article: {str(e)}") 