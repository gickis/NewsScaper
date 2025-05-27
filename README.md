# News Scraper and Analyzer

This project is a sophisticated news scraping and analysis system that leverages Generative AI to extract, summarize, and analyze news articles. It provides advanced semantic search capabilities to find relevant articles based on user queries.

## Features

- News article extraction from provided URLs
- AI-powered article summarization
- Topic identification using GenAI
- Advanced semantic search with query enhancement
- Vector database storage for efficient retrieval
- Command-line interface for easy usage

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd news-scraper
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
news-scraper/
├── src/
│   ├── scraper/         # News scraping functionality
│   ├── analyzer/        # AI analysis and summarization
│   ├── database/        # Vector database operations
│   └── cli.py          # Command-line interface
├── .env               # Environment variables
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Usage

### Command Line Interface

The tool can be used directly from the command line:

1. Scrape and analyze a single article:
```bash
python -m src.cli scrape "https://example.com/article"
```

2. Search for articles using semantic similarity:
```bash
python -m src.cli search "your search query" --limit 5
```

The search feature includes:
- Query enhancement using GenAI to understand context and synonyms
- Semantic matching of articles based on content and topics
- Relevance scoring for search results
- Option to disable query enhancement if needed

#### CLI Options

- `scrape` command:
  - `--print/--no-print`: Toggle printing article summary to console (default: print)

- `search` command:
  - `--limit`, `-l`: Maximum number of results to return (default: 5)
  - `--export`, `-e`: Export results to JSON file
  - `--no-enhance`: Disable query enhancement (use original query only)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 