import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import model

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['assets/style.css'])
server = app.server

# Define the layout components
app.layout = html.Div(className='container', children=[
    # Navigation component
    html.Div([
        html.P("Welcome to  Stock Dashboard!", className="start"),

        html.Div([
            # Stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range',
                start_date=dt(2024, 3, 10).date(),
                end_date=dt.now(),
                className='date-input'
            )
        ]),
        html.Div([
            # Stock price button
            html.Button('Get Stock Price', id='stock-price-button'),

            # Indicators button
            html.Button('Get Indicators', id='indicators-button'),
            ], className="selectors"),
        html.Div([
                # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),

            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
            ],className="forecast-input"),
        
    ], className="nav"),
    # Header component
    html.Div([
            html.Img(id='logo', className='logo'),
            html.H1(id='company-name', className='company-name'),
            html.P(id='description')
        ], className="header"),
    # Content component
    html.Div([
        html.Div(id="graphs-content"),
        html.Div(id="main-content"),
        html.Div(id="forecast-content")
    ], className="content")
])

# Callback to update the data based on the submitted stock code
@app.callback(
    [Output("description", "children"),
     Output("logo", "src"),
     Output("company-name", "children")],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):
    return model.update_data(n, val)

# Callback for displaying stock price graphs
@app.callback(
    Output("graphs-content", "children"),
    [Input("stock-price-button", "n_clicks"),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    return model.stock_price(n, start_date, end_date, val)

# Callback for displaying indicators
@app.callback(
    Output("main-content", "children"),
    [Input("indicators-button", "n_clicks"),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    return model.indicators(n, start_date, end_date, val)

# Callback for displaying forecast
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    return model.forecast(n, n_days, val)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
