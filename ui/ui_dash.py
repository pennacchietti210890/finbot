import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
import json
import requests
import random
import os
import math
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduced to 10 questions
EXAMPLE_QUESTIONS = [
    "How has Apple's stock performed over the last year?",
    "Compare the revenue growth of Microsoft and Google",
    "What are the key financial metrics for Tesla?",
    "What is Amazon's P/E ratio compared to the industry average?",
    "Show me the dividend yield for Coca-Cola",
    "Compare the P/E ratios of top 5 tech companies",
]

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Define the app's layout
app.layout = html.Div(
    [
        # Store components for maintaining state
        dcc.Store(id="chat-history-store", data={"messages": []}),
        dcc.Store(id="show-questions-store", data={"show": True}),
        dcc.Store(id="loading-store", data={"is_loading": False}),
        # Dummy div to avoid automatic triggering of callbacks
        html.Div(id="dummy-div", style={"display": "none"}),
        # Main container
        html.Div(
            [
                # Chat Interface Container
                html.Div(
                    [
                        # Chat Messages Area
                        html.Div(
                            id="chat-messages-container",
                            className="chat-messages-container",
                        ),
                        # Questions Area (stacked above the input)
                        html.Div(
                            # Pre-populate with empty div to prevent automatic execution
                            [html.Div(style={"display": "none"})],
                            id="questions-area", 
                            className="questions-area"
                        ),
                        # Loading indicator
                        html.Div(
                            [
                                html.Div(className="loading-circle"),
                                html.Div("Processing query...", className="loading-text"),
                            ],
                            id="loading-indicator",
                            className="loading-container",
                            style={"display": "none"},  # Initially hidden
                        ),
                        # Chat Input Area
                        html.Div(
                            [
                                dcc.Input(
                                    id="chat-input",
                                    type="text",
                                    placeholder="Ask about any S&P 500 company...",
                                    className="chat-input",
                                ),
                                html.Button(
                                    "Send",
                                    id="send-button",
                                    n_clicks=0,
                                    className="send-button",
                                ),
                            ],
                            className="chat-input-container",
                        ),
                    ],
                    className="chat-interface",
                )
            ],
            className="main-container",
        ),
    ]
)

# Custom CSS for the app
app.index_string = """
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
                margin-top: -15vh;
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
            
            /* Loading Container */
            .loading-container {
                z-index: 9999;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: rgba(0, 0, 0, 0.7);
                padding: 30px;
                border-radius: 15px;
                border: 2px solid rgba(0, 255, 136, 0.3);
                box-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
            }
            
            /* Loading Circle Animation */
            .loading-circle {
                width: 80px;
                height: 80px;
                margin: 0 auto 15px;
                border: 4px solid rgba(0, 255, 136, 0.1);
                border-top: 4px solid #00ff88;
                border-radius: 50%;
                animation: spin 1.5s linear infinite, pulse 2s infinite;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
                position: relative;
            }
            
            .loading-circle:before {
                content: '';
                position: absolute;
                top: -10px;
                left: -10px;
                right: -10px;
                bottom: -10px;
                border: 2px solid rgba(0, 255, 136, 0.2);
                border-radius: 50%;
                animation: pulse-out 2s infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 10px rgba(0, 255, 136, 0.4); }
                50% { box-shadow: 0 0 25px rgba(0, 255, 136, 0.7); }
                100% { box-shadow: 0 0 10px rgba(0, 255, 136, 0.4); }
            }
            
            @keyframes pulse-out {
                0% { transform: scale(1); opacity: 0.5; }
                50% { transform: scale(1.1); opacity: 0.2; }
                100% { transform: scale(1); opacity: 0.5; }
            }
            
            /* Loading Text */
            .loading-text {
                font-family: 'Orbitron', sans-serif;
                font-size: 18px;
                color: #00ff88;
                text-shadow: 0 0 10px rgba(0, 255, 136, 0.7);
                letter-spacing: 1px;
                animation: blink 1.5s infinite;
            }
            
            @keyframes blink {
                0% { opacity: 0.4; }
                50% { opacity: 1; }
                100% { opacity: 0.4; }
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
            
            /* Charts Container */
            .charts-container {
                margin-top: 15px;
                padding: 5px;
                border-radius: 6px;
                background-color: rgba(5, 15, 10, 0.5);
                border: 1px dashed rgba(0, 255, 136, 0.3);
                box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.3);
            }
            
            /* Style for chart title */
            .js-plotly-plot .gtitle {
                font-family: 'Orbitron', sans-serif !important;
                fill: #00ff88 !important;
                font-weight: 500 !important;
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
            
            /* Style for initial page (when questions are visible but chat is not) */
            .chat-interface:has(.questions-area:not(:empty)) .chat-input-container {
                margin-top: 20px;
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
"""


# Function to call backend API
def call_backend(query: str) -> Dict[str, Any]:
    """
    Send a request to the backend API and return the response.
    """
    try:
        logger.info(f"Sending request to backend at {API_URL}/chat")
        response = requests.post(
            f"{API_URL}/chat",
            json={"query": query},
            timeout=60  # Set a timeout of 60 seconds for longer queries
        )
        response.raise_for_status()
        
        logger.info(f"Received successful response from backend (status code: {response.status_code})")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Backend request timed out after 60 seconds")
        return {
            "text": "Sorry, the request took too long to process. Please try a simpler query or try again later.",
            "charts_data": "{}",
        }
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error when contacting backend at {API_URL}")
        return {
            "text": "Sorry, I couldn't connect to the backend server. Please check that the API is running.",
            "charts_data": "{}",
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error when calling backend: {str(e)}")
        return {
            "text": f"Sorry, I couldn't process your request due to a server error: {str(e)}",
            "charts_data": "{}",
        }
    except Exception as e:
        logger.error(f"Unexpected error when calling backend: {str(e)}")
        return {
            "text": "Sorry, an unexpected error occurred while processing your request.",
            "charts_data": "{}",
        }


# Function to parse response and extract charts
def parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the response from the backend and extract text and chart data.
    Returns a dictionary containing text and charts.
    """
    text = response.get("text", "")
    charts = []
    
    # Try to parse charts_data which is a JSON string
    charts_data = response.get("charts_data", "{}")
    
    try:
        # Parse the JSON string into a dictionary
        if charts_data and charts_data != "{}":
            logger.info(f"Parsing charts_data: {charts_data[:100]}...")
            stock_data = json.loads(charts_data)
            
            # Create a chart object in the format expected by generate_chart
            if "dates" in stock_data and "prices" in stock_data:
                logger.info(f"Creating chart with {len(stock_data['dates'])} data points")
                chart = {
                    "type": "line",
                    "title": "Stock Price Time Series",
                    "data": {
                        "Stock Price": {
                            "x": stock_data.get("dates", []),
                            "y": stock_data.get("prices", [])
                        }
                    }
                }
                charts.append(chart)
                logger.info("Chart added successfully")
            else:
                logger.warning(f"Missing 'dates' or 'prices' in stock_data: {list(stock_data.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing charts_data: {e}")
    except Exception as e:
        logger.error(f"Error processing chart data: {str(e)}")
    
    return {"text": text, "charts": charts}


# Initialize the question buttons
@app.callback(
    Output("questions-area", "children"),
    [Input("dummy-div", "children")],
    prevent_initial_call=False,
)
def initialize_question_buttons(_):
    buttons = []
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        button = html.Button(
            question,
            id={"type": "question-button", "index": i},
            className="question-button",
            n_clicks=0,
        )
        buttons.append(button)
    
    logger.info(f"Generated {len(buttons)} question buttons")
    return buttons


# Callback to show/hide questions based on chat history
@app.callback(
    Output("questions-area", "style"),
    [Input("chat-history-store", "data")],
)
def toggle_questions_visibility(chat_history):
    if chat_history and len(chat_history.get("messages", [])) > 0:
        return {"display": "none"}
    return {"display": "flex"}


# Callback to handle user input and generate responses - ONLY triggered by explicit button clicks
@app.callback(
    [
        Output("chat-messages-container", "children"),
        Output("chat-history-store", "data"),
        Output("chat-input", "value"),
        Output("chat-messages-container", "style"),
        Output("loading-store", "data"),
    ],
    [
        Input("send-button", "n_clicks"),
        Input({"type": "question-button", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("chat-input", "value"),
        State("chat-messages-container", "children"),
        State("chat-history-store", "data"),
        State("loading-store", "data"),
    ],
    prevent_initial_call=True,  # Critical: prevent callback on initial load
)
def update_chat(
    send_clicks,
    question_button_clicks,
    input_value,
    current_messages,
    chat_history,
    loading_data,
):
    # Check which input triggered the callback
    ctx_msg = dash.callback_context
    
    # Skip if no explicit trigger or during page load
    if not ctx_msg.triggered:
        logger.info("No trigger found, skipping callback")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the ID of the button that triggered the callback
    button_id = ctx_msg.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"Button clicked: {button_id}")
    
    # Initialize variables
    current_messages = current_messages or []
    chat_history = chat_history or {"messages": []}
    question = None
    
    # Determine the question based on which button was clicked
    if button_id == "send-button" and send_clicks and send_clicks > 0 and input_value:
        question = input_value
        logger.info(f"Send button clicked with input: {question}")
    elif "{" in button_id:  # This is a question button
        try:
            # Parse the button ID to get the index
            button_dict = json.loads(button_id.replace("'", "\""))
            if button_dict.get("type") == "question-button":
                index = button_dict.get("index", 0)
                if index < len(EXAMPLE_QUESTIONS) and question_button_clicks[index] > 0:
                    question = EXAMPLE_QUESTIONS[index]
                    logger.info(f"Question button {index} clicked: {question}")
        except Exception as e:
            logger.error(f"Error parsing button ID: {e}")
    
    # If no valid question, return unchanged
    if not question:
        logger.warning(f"No valid question found. Button ID: {button_id}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Update to show loading immediately - first update
    # (Just activating the loading indicator, not updating any other components yet)
    # Add user message to chat without waiting for API response
    user_message = html.Div(question, className="message user-message")
    updated_messages = current_messages + [user_message]
    initial_chat_display = {"display": "block"} if current_messages else {"display": "block"}
    
    # Immediate loading state update
    loading_data = {"is_loading": True}
    
    # Call backend for response
    logger.info(f"Calling backend API with query: {question}")
    response = call_backend(question)
    logger.info(f"Received response from backend: {str(response)[:100]}...")

    # Parse the response
    parsed_response = parse_response(response)
    logger.info(f"Parsed response with {len(parsed_response.get('charts', []))} charts")

    # Create chart components if any
    chart_components = []
    if parsed_response.get("charts"):
        chart_components = [
            generate_chart(chart_data) for chart_data in parsed_response["charts"]
        ]
        logger.info(f"Generated {len(chart_components)} chart components")

    # Add assistant message to chat
    assistant_message = html.Div(
        [
            html.P(parsed_response["text"]),
            # Add charts container if there are charts
            html.Div(
                children=chart_components,
                className="charts-container",
            ) if chart_components else None,
        ],
        className="message assistant-message",
    )

    updated_messages = updated_messages + [assistant_message]

    # Update chat history
    chat_history["messages"].append({"role": "user", "content": question})
    chat_history["messages"].append(
        {
            "role": "assistant",
            "content": parsed_response["text"],
            "charts": parsed_response.get("charts", []),
        }
    )
    
    # Hide loading indicator
    loading_data = {"is_loading": False}

    # Show chat messages container
    logger.info("Returning updated chat interface")
    return (
        updated_messages, 
        chat_history, 
        "", 
        {"display": "block"},
        loading_data,
    )


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
    
    logger.info(f"Generating chart of type: {chart_type}, title: {title}")

    fig = go.Figure()

    if chart_type == "line":
        # Line chart for time series data
        for series_name, series_data in data.items():
            x_data = series_data.get("x", [])
            y_data = series_data.get("y", [])
            
            # Validate data to ensure we have matching x and y data points
            if x_data and y_data and len(x_data) == len(y_data):
                logger.info(f"Adding line series '{series_name}' with {len(x_data)} data points")
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data, mode="lines", name=series_name)
                )
            else:
                logger.warning(f"Invalid data for series {series_name}: x={len(x_data)}, y={len(y_data)}")

    elif chart_type == "bar":
        # Bar chart for comparison data
        categories = list(data.keys())
        values = [data[category] for category in categories]
        if categories and values:
            logger.info(f"Adding bar chart with {len(categories)} categories")
            fig.add_trace(go.Bar(x=categories, y=values))
        else:
            logger.warning("Empty data for bar chart")

    elif chart_type == "pie":
        # Pie chart for distribution data
        labels = list(data.keys())
        values = [data[label] for label in labels]
        if labels and values:
            logger.info(f"Adding pie chart with {len(labels)} segments")
            fig.add_trace(go.Pie(labels=labels, values=values))
        else:
            logger.warning("Empty data for pie chart")

    # Apply dark theme to charts
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,10,0.7)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    logger.info("Chart generation completed")
    return dcc.Graph(
        figure=fig, config={"displayModeBar": False}, style={"margin": "20px 0"}
    )


# Add client-side callback for loading status
app.clientside_callback(
    """
    function(loadingData) {
        console.log("Loading data changed:", loadingData);
        
        // Get the loading element
        var loadingElement = document.getElementById('loading-indicator');
        
        if (loadingData && loadingData.is_loading === true) {
            console.log("Setting loading display to block");
            
            // Ensure the element exists and update its style directly
            if (loadingElement) {
                loadingElement.style.display = 'block';
            }
            
            return {"display": "block"};
        } else {
            console.log("Setting loading display to none");
            
            // Ensure the element exists and update its style directly
            if (loadingElement) {
                loadingElement.style.display = 'none';
            }
            
            return {"display": "none"};
        }
    }
    """,
    Output("loading-indicator", "style"),
    Input("loading-store", "data"),
)

# Add client-side callback for auto-scrolling chat messages
app.clientside_callback(
    """
    function(children) {
        if (children) {
            setTimeout(function() {
                var chatContainer = document.getElementById('chat-messages-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-messages-container", "children", allow_duplicate=True),
    Input("chat-messages-container", "children"),
    prevent_initial_call=True,
)

# Add client-side callbacks to immediately show loading when buttons are clicked
app.clientside_callback(
    """
    function(n_clicks) {
        console.log("Send button clicked, showing loading indicator");
        
        // Update the store to trigger the loading indicator
        if (n_clicks > 0) {
            // Get the loading element and show it immediately
            var loadingElement = document.getElementById('loading-indicator');
            if (loadingElement) {
                loadingElement.style.display = 'block';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("send-button", "n_clicks", allow_duplicate=True),
    Input("send-button", "n_clicks"),
    prevent_initial_call=True,
)

# Add client-side callback for question buttons to show loading immediately
app.clientside_callback(
    """
    function(n_clicks_list, _) {
        console.log("Question button clicked, checking which one", n_clicks_list);
        
        // Check if any button has been clicked
        if (n_clicks_list && n_clicks_list.some(clicks => clicks > 0)) {
            console.log("Question button was clicked, showing loading indicator");
            
            // Get the loading element and show it immediately
            var loadingElement = document.getElementById('loading-indicator');
            if (loadingElement) {
                loadingElement.style.display = 'block';
                console.log("Set loading indicator display to block from question button");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output({"type": "question-button", "index": dash.ALL}, "n_clicks", allow_duplicate=True),
    Input({"type": "question-button", "index": dash.ALL}, "n_clicks"),
    State("dummy-div", "children"),
    prevent_initial_call=True,
)

# Add a separate client-side callback to help debug the loading indicator visibility
app.clientside_callback(
    """
    function() {
        // This runs on page load
        console.log("Setting up loading indicator visibility monitor");
        
        // Set up a MutationObserver to monitor the loading indicator's style changes
        var loadingElement = document.getElementById('loading-indicator');
        if (loadingElement) {
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.attributeName === 'style') {
                        console.log("Loading indicator style changed:", 
                                    loadingElement.style.display);
                    }
                });
            });
            
            observer.observe(loadingElement, { attributes: true });
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-div", "children", allow_duplicate=True),
    Input("dummy-div", "children"),
    prevent_initial_call=True,
)


def main():
    """Entry point for running the Dash application"""
    app.run(debug=False, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    main()
