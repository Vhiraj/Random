```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import collections
import random
import datetime
import threading
import time
import pandas as pd # For easier data handling in the callback

# --- Explanation Block ---
"""
Interactive Dashboard for Real-time Sensor Data

This Python script creates an interactive web-based dashboard using Plotly Dash
to visualize simulated real-time sensor data. It's designed to be a standalone
application, suitable for demonstrating real-time data streaming and visualization.

Key Features:
1.  **Sensor Data Simulation**: A dedicated thread continuously generates
    mock sensor readings (Temperature, Humidity, Pressure) with precise
    timestamps. This mimics data acquisition from physical sensors in a
    real-world scenario.
2.  **In-memory Data Buffer**: A `collections.deque` is employed as a highly
    efficient, fixed-size circular buffer to store a configurable number of
    the most recent sensor data points in memory. This ensures the dashboard
    only displays relevant, fresh data.
3.  **Thread-Safe Data Access**: A `threading.Lock` is implemented to
    guarantee thread-safe access to the shared data buffer (`sensor_data_queue`),
    preventing race conditions between the data simulation thread and the
    Dash application's data retrieval callback.
4.  **Plotly Dash Framework**: Leverages the powerful Plotly Dash library
    (built on Flask and React.js) for constructing a responsive, interactive,
    and modern web dashboard interface.
5.  **Real-time Visualization**: The dashboard automatically updates its
    visualizations (line graphs) at a specified interval, fetching the
    latest data from the buffer and redrawing the plots seamlessly.
6.  **Interactive Graphs**: Plotly's native interactivity allows users to
    zoom, pan, and hover over data points to inspect values, enhancing data
    exploration capabilities. `uirevision` is used to maintain zoom/pan state.
7.  **Structured and Readable Code**: The project demonstrates best practices
    through clear variable naming, comprehensive docstrings, inline comments,
    and a modular organization within a single file, making it easy to
    understand, maintain, and extend.

To run this dashboard:
1.  Ensure you have Python 3.7+ installed.
2.  Install the necessary Python libraries using pip:
    `pip install dash plotly pandas`
3.  Save this code as a Python file (e.g., `realtime_sensor_dashboard.py`).
4.  Execute the script from your terminal:
    `python realtime_sensor_dashboard.py`
5.  Open your web browser and navigate to the local address provided in the
    terminal output (typically `http://127.0.0.1:8050/`).
"""

# --- Global Configuration and Data Storage ---

MAX_DATA_POINTS = 150  # Maximum number of data points to retain and display in the graphs
DASHBOARD_UPDATE_INTERVAL_MS = 1000  # Dashboard refresh rate in milliseconds (1 second)
SENSOR_SIMULATION_INTERVAL_SEC = 0.5  # Frequency at which new sensor data is generated (0.5 seconds)

# A thread-safe, fixed-size queue to store recent sensor data.
# Each element is a dictionary: {'timestamp': datetime, 'temp': float, 'humidity': float, 'pressure': float}
sensor_data_queue = collections.deque(maxlen=MAX_DATA_POINTS)
data_lock = threading.Lock() # Lock to ensure safe concurrent access to `sensor_data_queue`

# Defines the properties for each simulated sensor, including base value,
# random fluctuation range, and units for display.
SENSOR_PROPERTIES = {
    'Temperature': {'base': 25.0, 'range': (-2.0, 2.0), 'unit': '°C', 'color': '#1f77b4'}, # Blue
    'Humidity': {'base': 60.0, 'range': (-5.0, 5.0), 'unit': '%', 'color': '#2ca02c'},     # Green
    'Pressure': {'base': 1012.0, 'range': (-10.0, 10.0), 'unit': 'hPa', 'color': '#ff7f0e'} # Orange
}

# --- Sensor Data Simulation Function ---

def simulate_sensor_data() -> None:
    """
    Continuously generates mock sensor data at a fixed interval and adds it
    to the global `sensor_data_queue`. This function is intended to run in a
    separate thread to mimic real-time data streams.
    """
    print("INFO: Starting sensor data simulation thread...")
    while True:
        timestamp = datetime.datetime.now()
        new_data = {'timestamp': timestamp}

        for sensor_name, props in SENSOR_PROPERTIES.items():
            # Generate a value by adding a random fluctuation to the base value
            value = props['base'] + random.uniform(*props['range'])
            new_data[sensor_name.lower()] = round(value, 2) # Store with lowercase key for consistency

        # Acquire lock before modifying the shared queue
        with data_lock:
            sensor_data_queue.append(new_data)
        
        # Optional: Print generated data to console for debugging
        # print(f"DEBUG: Generated data at {new_data['timestamp'].strftime('%H:%M:%S.%f')[:-3]}: "
        #       f"Temp={new_data['temperature']:.2f}, Hum={new_data['humidity']:.2f}, "
        #       f"Press={new_data['pressure']:.2f}")
        
        time.sleep(SENSOR_SIMULATION_INTERVAL_SEC)

# --- Dash App Initialization ---

# Initialize the Dash application.
# `__name__` is used to let Dash know where to find static assets.
# `title` sets the browser tab title, `update_title` sets text during updates.
app = dash.Dash(__name__,
                title="Real-time Sensor Dashboard",
                update_title='Updating...')

# --- Dashboard Layout ---

def create_dashboard_layout() -> html.Div:
    """
    Constructs the HTML layout for the Dash dashboard.
    This includes the main title, a dynamic update message, and three
    graph components for each sensor, along with the interval component
    that triggers updates.
    """
    return html.Div(
        style={
            'fontFamily': 'Arial, sans-serif',
            'maxWidth': '1200px',
            'margin': 'auto',
            'padding': '20px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '8px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
        },
        children=[
            html.H1(
                "Real-time Sensor Data Dashboard",
                style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px', 'fontSize': '2.5em'}
            ),
            html.Div(
                id='live-update-text',
                style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '1.2em', 'color': '#555', 'fontWeight': 'bold'}
            ),
            html.Div(
                className="graph-container-row",
                style={
                    'display': 'flex',
                    'flexWrap': 'wrap', # Allows graphs to wrap to the next line on smaller screens
                    'justifyContent': 'space-around',
                    'gap': '20px' # Space between graph cards
                },
                children=[
                    html.Div(
                        className="graph-card",
                        style={
                            'flex': '1 1 30%', # Allows flexible sizing, minimum 30% width
                            'minWidth': '320px', # Minimum width before wrapping
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'borderRadius': '8px',
                            'backgroundColor': '#fff',
                            'padding': '15px',
                            'boxSizing': 'border-box' # Include padding in width calculation
                        },
                        children=dcc.Graph(id='temperature-graph', config={'displayModeBar': False})
                    ),
                    html.Div(
                        className="graph-card",
                        style={
                            'flex': '1 1 30%',
                            'minWidth': '320px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'borderRadius': '8px',
                            'backgroundColor': '#fff',
                            'padding': '15px',
                            'boxSizing': 'border-box'
                        },
                        children=dcc.Graph(id='humidity-graph', config={'displayModeBar': False})
                    ),
                    html.Div(
                        className="graph-card",
                        style={
                            'flex': '1 1 30%',
                            'minWidth': '320px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'borderRadius': '8px',
                            'backgroundColor': '#fff',
                            'padding': '15px',
                            'boxSizing': 'border-box'
                        },
                        children=dcc.Graph(id='pressure-graph', config={'displayModeBar': False})
                    ),
                ]
            ),
            # `dcc.Interval` component triggers the callback at a set interval
            dcc.Interval(
                id='interval-component',
                interval=DASHBOARD_UPDATE_INTERVAL_MS, # in milliseconds
                n_intervals=0 # Initial value, increments automatically
            )
        ]
    )

# Assign the created layout to the Dash application
app.layout = create_dashboard_layout()

# --- Dashboard Callbacks for Real-time Updates ---

@app.callback(
    Output('temperature-graph', 'figure'),
    Output('humidity-graph', 'figure'),
    Output('pressure-graph', 'figure'),
    Output('live-update-text', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n_intervals: int) -> tuple[go.Figure, go.Figure, go.Figure, html.Span]:
    """
    This callback function is triggered periodically by the `dcc.Interval` component.
    It reads the latest sensor data from the shared queue, processes it, and
    updates the three Plotly graphs and the "Last updated" text.

    Args:
        n_intervals (int): The number of times the `dcc.Interval` component
                           has fired since the app started.

    Returns:
        tuple[go.Figure, go.Figure, go.Figure, html.Span]: A tuple containing
        the Plotly figures for Temperature, Humidity, and Pressure, respectively,
        followed by an HTML `Span` element displaying the last update time.
    """
    
    # Read data from the shared queue in a thread-safe manner
    # A copy of the deque's content is made to avoid holding the lock
    # for the duration of data processing and plotting.
    with data_lock:
        current_data = list(sensor_data_queue) 

    if not current_data:
        # If no data has been generated yet, return placeholder figures and message
        empty_fig_layout = go.Layout(
            title="Waiting for sensor data...",
            xaxis_title="Time",
            yaxis_title="Value",
            uirevision='no update', # Prevents graph from resetting zoom/pan when data arrives
            height=250, # Fixed height for empty state
            margin={'l': 40, 'r': 10, 't': 40, 'b': 30}
        )
        empty_fig = go.Figure(layout=empty_fig_layout)
        last_update_info = html.Span("No data received yet. Simulating...", style={'color': '#888'})
        return empty_fig, empty_fig, empty_fig, last_update_info

    # Convert the list of dictionaries into a Pandas DataFrame for efficient
    # data manipulation and plotting.
    df = pd.DataFrame(current_data)
    df = df.set_index('timestamp') # Set timestamp as index for time series plotting
    df.sort_index(inplace=True) # Ensure data is sorted by timestamp

    # Prepare the last update timestamp for display on the dashboard
    last_timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    last_update_info_text = f"Last updated: {last_timestamp}"

    # Generate a Plotly figure for each sensor
    figures = {}
    for sensor_name, props in SENSOR_PROPERTIES.items():
        key = sensor_name.lower() # Column name in DataFrame
        
        # Define a consistent Y-axis range for each sensor for better visualization
        # The range is extended slightly beyond the simulation's min/max to prevent
        # data points from touching the edges and to provide visual breathing room.
        y_axis_min = props['base'] + props['range'][0] - 5
        y_axis_max = props['base'] + props['range'][1] + 5

        # Create the Plotly Graph Object figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df.index,
                    y=df[key],
                    mode='lines+markers', # Display both lines and markers
                    name=sensor_name,
                    line=dict(width=2, color=props['color']) # Use predefined color
                )
            ],
            layout=go.Layout(
                title=f"{sensor_name} Data Over Time",
                xaxis=dict(
                    title="Time",
                    rangeslider=dict(visible=True), # Allow users to select time range
                    type="date", # Treat x-axis as date/time
                    rangeselector=dict( # Quick selection buttons for time ranges
                        buttons=list([
                            dict(count=1, label="1m", step="minute", stepmode="backward"),
                            dict(count=5, label="5m", step="minute", stepmode="backward"),
                            dict(count=10, label="10m", step="minute", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                yaxis=dict(
                    title=f"{sensor_name} {props['unit']}",
                    range=[y_axis_min, y_axis_max], # Fixed Y-axis range
                    autorange=False # Disable auto-ranging on Y-axis
                ),
                margin={'l': 40, 'r': 10, 't': 40, 'b': 30}, # Adjust plot margins
                hovermode='x unified', # Show all data points at a given x-value on hover
                uirevision='no update', # Crucial: Preserves user's zoom/pan state on updates
                transition_duration=500, # Smooth transition animation for data updates
                height=250 # Consistent height for all graph cards
            )
        )
        figures[sensor_name] = fig
    
    # Return the three generated figures and the updated text for the dashboard
    return (
        figures['Temperature'],
        figures['Humidity'],
        figures['Pressure'],
        html.Span(last_update_info_text, style={'color': '#007bff'}) # Stylish last update message
    )

# --- Main Execution Block ---

if __name__ == '__main__':
    # Start the sensor data simulation in a separate, daemonized thread.
    # A daemon thread will automatically terminate when the main program exits,
    # preventing it from running indefinitely in the background if the Dash app closes.
    sensor_thread = threading.Thread(target=simulate_sensor_data, daemon=True)
    sensor_thread.start()

    print(f"INFO: Dash app starting...")
    print(f"INFO: Open your web browser to http://127.0.0.1:8050/")
    # Run the Dash app server.
    # `debug=True` enables hot-reloading and debug tools (set to `False` for production).
    # `host='0.0.0.0'` makes the server accessible from other devices on the network;
    # use `host='127.0.0.1'` to restrict access to the local machine only.
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```