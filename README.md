# FinBot - Financial Information Chatbot

FinBot is an AI-powered chatbot that provides information about publicly listed companies, primarily from the S&P 500 index. Users can ask questions in natural language and receive AI-generated responses with relevant charts and financial insights.

## Project Structure

```
finbot/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend application
│   ├── llm_service.py   # LLM integration service
│   └── llm/             # LLM-related utilities
├── ui/
│   ├── __init__.py
│   ├── ui_dash.py       # Dash application with cyberpunk UI
├── pyproject.toml       # Poetry configuration and dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Setup

### Prerequisites

- Python 3.13 or higher
- [Poetry](https://python-poetry.org/docs/#installation) dependency manager
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd finbot
   ```

2. Create environment files:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file to add your OpenAI API key.

4. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Running the Application

You can run the application components using Poetry's run commands:

### Running the Backend

```bash
poetry run start-backend
```

The API will be available at `http://localhost:8000`.

### Running the Frontend

```bash
poetry run start-frontend
```

The Dash app will be available at `http://localhost:8502`.

## Development with Poetry

To enter the Poetry virtual environment for development:

```
poetry shell
```

To add new dependencies:

```
poetry add package-name
```

To add development-only dependencies:

```
poetry add --group dev package-name
```

## Usage

1. Open the app in your browser at `http://localhost:8502`.
2. You'll see a cyberpunk-styled interface with example questions displayed.
3. Click on any example question or type your own query about a publicly listed company.
4. The chat interface will appear with the AI-generated response and any relevant charts.
5. Continue your conversation with follow-up questions.

## Frontend Features

The modern Dash-based frontend includes:
- Cyberpunk-inspired user interface with neon green styling
- Sleek dark mode design with futuristic typography
- Clickable example questions that display above the chat input
- Hidden chat container that appears after first interaction
- Responsive design that works on different screen sizes
- Integrated Plotly charts with interactive visualizations
- Custom animations and visual effects for interactive elements

## Example Questions

- How has Apple performed over the last year?
- What are Tesla's key financial metrics?
- Compare the revenue growth of Microsoft and Google.
- What is the P/E ratio of Amazon compared to industry average?
- Show me the dividend yield for Coca-Cola
- What sectors are performing best in the current market?

## Features

- Natural language processing for financial queries
- Integration with financial data sources
- Dynamic chart generation for data visualization
- Conversational UI with persistent chat history
- Modern, cyberpunk-styled interface
- Seamless integration between Python backend and frontend

## Limitations

- Currently only supports companies in the S&P 500 index
- Data may not be real-time and should not be used for investment decisions
- Limited to financial information that is publicly available 