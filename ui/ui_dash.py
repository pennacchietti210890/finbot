import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
import json
import requests
import random
import os
import math
from typing import Dict, Any, List

# Reduced to 10 questions
EXAMPLE_QUESTIONS = [
    "How has Apple's stock performed over the last year?",
    "Compare the revenue growth of Microsoft and Google",
    "What are the key financial metrics for Tesla?",
    "What is Amazon's P/E ratio compared to the industry average?",
    "Show me the dividend yield for Coca-Cola",
    "What sectors are performing best in the current market?",
    "Compare the P/E ratios of top 5 tech companies",
]

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize the Dash app
app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Define the app's layout
app.layout = html.Div([
    # Store components for maintaining state
    dcc.Store(id='chat-history-store', data={"messages": []}),
    dcc.Store(id='show-questions-store', data={"show": True}),
    
    # Main container
    html.Div([
        # Chat Interface Container
        html.Div([
            # Chat Messages Area
            html.Div(id='chat-messages-container', className='chat-messages-container'),
            
            # Questions Area (stacked above the input)
            html.Div(id='questions-area', className='questions-area'),
            
            # Chat Input Area
            html.Div([
                dcc.Input(
                    id='chat-input',
                    type='text',
                    placeholder='Ask about any S&P 500 company...',
                    className='chat-input'
                ),
                html.Button('Send', id='send-button', n_clicks=0, className='send-button')
            ], className='chat-input-container')
        ], className='chat-interface')
    ], className='main-container')
])

# Custom CSS for the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>FinBot</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global Styles */
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Rajdhani:wght@300;400;500;600&display=swap');
            
            body {
                margin: 0;
                padding: 0;
                background-color: #000000;
                color: white;
                font-family: 'Rajdhani', sans-serif;
                height: 100vh;
                overflow: hidden;
                background-image: radial-gradient(circle at 50% 50%, rgba(17, 38, 70, 0.2) 0%, rgba(0, 0, 0, 1) 100%);
            }
            
            /* Main Container */
            .main-container {
                width: 100%;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            }
            
            /* Chat Interface */
            .chat-interface {
                width: 60%;
                max-width: 800px;
                height: 80vh;
                padding: 0;
                z-index: 10;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                position: relative;
                margin-bottom: 0;
            }
            
            /* Chat Messages Container */
            .chat-messages-container {
                flex-grow: 1;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 10px;
                background-color: rgba(10, 10, 10, 0.1);
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
                max-height: none;
                border: 1px solid rgba(0, 255, 136, 0.2);
                box-shadow: 0 0 15px rgba(0, 255, 136, 0.1);
                /* Initially hide the container */
                display: none;
            }
            
            /* Scrollbar styling */
            .chat-messages-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-messages-container::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 3px;
            }
            
            .chat-messages-container::-webkit-scrollbar-thumb {
                background: rgba(0, 255, 136, 0.4);
                border-radius: 3px;
            }
            
            /* Chat Message Styles */
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 8px;
                max-width: 80%;
                background-color: rgba(10, 10, 10, 0.7);
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                font-family: 'Rajdhani', sans-serif;
                font-weight: 500;
            }
            
            .user-message {
                background-color: rgba(0, 60, 30, 0.7);
                margin-left: auto;
                text-align: right;
                border-bottom-right-radius: 2px;
                border-left: 3px solid #00ff88;
                color: #7fffcf; /* Bright neon green text */
                text-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
                box-shadow: 0 2px 10px rgba(0, 255, 136, 0.2);
            }
            
            .assistant-message {
                background-color: rgba(20, 30, 20, 0.7);
                margin-right: auto;
                border-bottom-left-radius: 2px;
                border-right: 2px solid #00ff88;
            }
            
            /* Questions Area */
            .questions-area {
                margin: 0 auto 15px auto;
                padding: 10px;
                text-align: center;
                z-index: 5;
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 8px;
                max-width: 90%;
                min-height: 0;
            }
            
            .question-button {
                padding: 8px 14px;
                border: none;
                border-radius: 4px;
                background-color: rgba(10, 35, 20, 0.75);
                color: #00ff88;
                font-family: 'Orbitron', sans-serif;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 12px;
                letter-spacing: 0.5px;
                font-weight: 500;
                box-shadow: 0 0 8px rgba(0, 255, 136, 0.2), inset 0 0 3px rgba(0, 255, 136, 0.2);
                white-space: normal;
                margin: 4px 2px;
                border-bottom: 2px solid #00aa44;
                border-top: 1px solid rgba(0, 255, 136, 0.3);
                text-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
                text-transform: uppercase;
            }
            
            .question-button:hover {
                background-color: rgba(10, 50, 30, 0.9);
                color: #80ffaa;
                transform: translateY(-2px);
                box-shadow: 0 0 12px rgba(0, 255, 136, 0.4), inset 0 0 8px rgba(0, 255, 136, 0.3);
                border-bottom: 2px solid #00ff88;
                text-shadow: 0 0 10px rgba(0, 255, 136, 0.7);
            }
            
            /* Chat Input Container */
            .chat-input-container {
                display: flex;
                gap: 10px;
                z-index: 20;
                margin-bottom: 10px;
            }
            
            .chat-input {
                flex-grow: 1;
                padding: 14px 18px;
                border-radius: 4px;
                border: 1px solid rgba(0, 255, 136, 0.3);
                background-color: rgba(10, 25, 15, 0.7);
                color: white;
                font-size: 15px;
                font-family: 'Rajdhani', sans-serif;
                font-weight: 500;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.1), inset 0 0 5px rgba(0, 0, 0, 0.3);
                transition: all 0.3s;
                letter-spacing: 0.5px;
            }
            
            .chat-input:focus {
                outline: none;
                background-color: rgba(15, 35, 25, 0.8);
                box-shadow: 0 0 15px rgba(0, 255, 136, 0.2), inset 0 0 5px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(0, 255, 136, 0.5);
            }
            
            .send-button {
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                background-color: #153450;
                color: #00ffaa;
                cursor: pointer;
                transition: all 0.3s;
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
                font-size: 14px;
                letter-spacing: 1px;
                text-transform: uppercase;
                box-shadow: 0 0 10px rgba(0, 255, 170, 0.2), inset 0 0 5px rgba(0, 255, 170, 0.1);
                border-bottom: 3px solid #007744;
                text-shadow: 0 0 5px rgba(0, 255, 170, 0.5);
                position: relative;
                overflow: hidden;
            }
            
            .send-button:before {
                content: '';
                position: absolute;
                top: -10px;
                left: -10px;
                width: calc(100% + 20px);
                height: calc(100% + 20px);
                background: linear-gradient(45deg, transparent 20%, rgba(0, 255, 170, 0.1), transparent 80%);
                animation: shimmer 3s infinite;
                z-index: -1;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%) translateY(100%); }
                100% { transform: translateX(100%) translateY(-100%); }
            }
            
            .send-button:hover {
                background-color: #1a4565;
                color: #7fffcf;
                box-shadow: 0 0 15px rgba(0, 255, 170, 0.4), inset 0 0 10px rgba(0, 255, 170, 0.2);
                transform: translateY(-2px);
                border-bottom: 3px solid #00aa66;
                text-shadow: 0 0 8px rgba(0, 255, 170, 0.7);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Function to call backend API
def call_backend(query: str) -> Dict[str, Any]:
    """
    Send a request to the backend API and return the response.
    """
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"query": query},
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"text": f"Sorry, I couldn't process your request due to a server error: {str(e)}", "charts": []}

# Function to parse response and extract charts
def parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the response from the backend and extract any charts.
    Returns a tuple of (text, charts)
    """
    text = response.get("text", "")
    charts = response.get("charts", [])
    
    # If charts is an empty list but there are chart objects in the response
    # Try to extract them from the response format
    if not charts:
        # Check if there's a 'data' key that might contain charts
        if "data" in response:
            data = response["data"]
            if isinstance(data, dict) and "charts" in data:
                charts = data["charts"]
            elif isinstance(data, list):
                # Assume the list contains chart objects
                charts = data
    
    return {
        "text": text,
        "charts": charts
    }



# Callback to generate stacked question buttons
@app.callback(
    Output('questions-area', 'children'),
    [Input('chat-history-store', 'data'),
     Input('show-questions-store', 'data')]
)
def generate_question_buttons(chat_history, show_questions_data):
    # Hide questions if user has submitted a query or if show_questions is False
    if (chat_history and len(chat_history.get("messages", [])) > 0) or not show_questions_data.get("show", True):
        return []
    
    # Create buttons for each example question
    buttons = []
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        button = html.Button(
            question,
            id={'type': 'question-button', 'index': i},
            className='question-button',
            n_clicks=0
        )
        buttons.append(button)
    
    return buttons

# Callback to handle user input and generate responses
@app.callback(
    [Output('chat-messages-container', 'children'),
     Output('chat-history-store', 'data'),
     Output('chat-input', 'value'),
     Output('show-questions-store', 'data'),
     Output('chat-messages-container', 'style')],
    [Input('send-button', 'n_clicks'),
     Input({'type': 'question-button', 'index': dash.ALL}, 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-messages-container', 'children'),
     State('chat-history-store', 'data'),
     State('show-questions-store', 'data')]
)
def update_chat(send_clicks, question_button_clicks, 
                input_value, current_messages, chat_history, show_questions_data):
    current_messages = current_messages or []
    chat_history = chat_history or {"messages": []}
    
    # Get which input triggered the callback
    trigger = ctx.triggered_id
    
    # Determine the question from question button or input
    question = None
    
    # Check if triggered by a question button
    if isinstance(trigger, dict) and trigger.get('type') == 'question-button':
        index = trigger.get('index', 0)
        if index < len(EXAMPLE_QUESTIONS):
            question = EXAMPLE_QUESTIONS[index]
    # Check if triggered by send button with input text
    elif trigger == 'send-button' and input_value:
        question = input_value
    
    # If no valid question, return unchanged
    if not question:
        return current_messages, chat_history, input_value, show_questions_data, {'display': 'none'}
    
    # Add user message to chat
    user_message = html.Div(question, className='message user-message')
    updated_messages = current_messages + [user_message]
    
    # Call backend for response
    response = call_backend(question)
    
    # Parse the response
    parsed_response = parse_response(response)
    
    # Add assistant message to chat
    assistant_message = html.Div([
        html.P(parsed_response["text"]),
        # Add any charts if available
        html.Div(id={'type': 'charts-container', 'index': len(chat_history["messages"])},
                children=[generate_chart(chart_data) for chart_data in parsed_response.get("charts", [])])
    ], className='message assistant-message')
    
    updated_messages = updated_messages + [assistant_message]
    
    # Update chat history
    chat_history["messages"].append({
        "role": "user",
        "content": question
    })
    chat_history["messages"].append({
        "role": "assistant",
        "content": parsed_response["text"],
        "charts": parsed_response.get("charts", [])
    })
    
    # Hide questions after user input
    show_questions_data = {"show": False}
    
    # Show chat messages container
    return updated_messages, chat_history, "", show_questions_data, {'display': 'block'}

# Function to generate a Plotly chart from data
def generate_chart(chart_data):
    """
    Generate a Plotly chart from the data returned by the backend.
    The backend can return different types of charts:
    - line: for time series data
    - bar: for comparison data
    - pie: for distribution data
    """
    chart_type = chart_data.get("type", "line")
    title = chart_data.get("title", "")
    data = chart_data.get("data", {})
    
    fig = go.Figure()
    
    if chart_type == "line":
        # Line chart for time series data
        for series_name, series_data in data.items():
            x_data = series_data.get("x", [])
            y_data = series_data.get("y", [])
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name=series_name
                )
            )
    
    elif chart_type == "bar":
        # Bar chart for comparison data
        categories = list(data.keys())
        values = [data[category] for category in categories]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values
            )
        )
    
    elif chart_type == "pie":
        # Pie chart for distribution data
        labels = list(data.keys())
        values = [data[label] for label in labels]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values
            )
        )
    
    # Apply dark theme to charts
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,10,0.7)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dcc.Graph(
        figure=fig, 
        config={'displayModeBar': False},
        style={"margin": "20px 0"}
    )

def main():
    """Entry point for running the Dash application"""
    app.run(debug=False, host='0.0.0.0', port=8502) 

if __name__ == '__main__':
    main()