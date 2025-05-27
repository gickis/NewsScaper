import click
import json
from typing import List, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime
from .scraper.article_scraper import ArticleScraper
from .analyzer.article_analyzer import ArticleAnalyzer
from .database.vector_store import VectorStore

# Initialize components
scraper = ArticleScraper()
analyzer = ArticleAnalyzer()
vector_store = VectorStore()

def print_article_summary(article_data: Dict[str, Any], analysis: Any) -> None:
    """Print article summary and analysis results."""
    click.echo("\nArticle Summary:")
    click.echo("-" * 50)
    click.echo(f"Title: {article_data['title']}")
    click.echo(f"Summary: {analysis.summary}")
    click.echo("\nTopics:")
    for topic in analysis.topics:
        click.echo(f"- {topic}")
    click.echo("\nKey Points:")
    for point in analysis.key_points:
        click.echo(f"- {point}")

def print_search_results(results: List[Dict[str, Any]]) -> None:
    """Print search results in a formatted way."""
    click.echo("\nSearch Results:")
    click.echo("-" * 50)
    
    # Show enhanced query if it was used (only once at the beginning)
    if results and results[0].get('enhanced_query'):
        click.echo(f"Enhanced query: {results[0]['enhanced_query']}\n")
    
    for i, article in enumerate(results, 1):
        click.echo(f"\n{i}. {article['title']}")
        click.echo(f"Summary: {article['summary']}")
        click.echo("Topics: " + ", ".join(article['topics']))
        if 'similarity' in article:
            click.echo(f"Relevance: {1 - article['similarity']:.2%}")
        click.echo("-" * 30)

@click.group()
def cli():
    """News Scraper and Analyzer CLI tool."""
    pass

@cli.command()
@click.argument('url')
@click.option('--print/--no-print', default=True, help='Print article summary to console')
def scrape(url: str, print: bool) -> None:
    """Scrape and analyze an article from the provided URL."""
    try:
        click.echo(f"Scraping article from: {url}")
        
        # Scrape the article
        article_data = scraper.scrape_article(url)
        
        # Analyze the article
        analysis = analyzer.analyze_article(
            title=article_data['title'],
            text=article_data['text']
        )
        
        # Generate unique ID and store article
        article_id = str(uuid.uuid4())
        vector_store.add_article(
            article_id=article_id,
            title=article_data['title'],
            text=article_data['text'],
            summary=analysis.summary,
            topics=analysis.topics,
            metadata={
                'url': url,
                'publish_date': article_data['publish_date'],
                'scraped_at': datetime.now().isoformat()
            }
        )
        
        click.echo(f"Article stored with ID: {article_id}")
        
        if print:
            print_article_summary(article_data, analysis)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('query')
@click.option('--limit', '-l', help='Maximum number of results', default=5)
@click.option('--export', '-e', help='Export results to JSON file')
@click.option('--no-enhance', is_flag=True, help='Disable query enhancement')
def search(query: str, limit: int, export: str, no_enhance: bool) -> None:
    """Search for articles using semantic similarity with enhanced query understanding."""
    try:
        click.echo(f"Searching for: {query}")
        
        # Search articles with enhanced query understanding
        results = vector_store.search_articles(
            query=query,
            n_results=limit,
            enhance_query=not no_enhance
        )
        
        print_search_results(results)
        
        # Export to file if requested
        if export:
            with open(export, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            click.echo(f"\nResults exported to: {export}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 