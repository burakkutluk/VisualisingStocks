import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def update_data(n, val):
    if n is None or val is None:
        raise PreventUpdate
    try:
        ticker = yf.Ticker(val)
        info = ticker.info
        name = info.get("longName", None)
        logo_url = info.get("logo_url", None)
        description = info.get("longBusinessSummary", None)
        return description, logo_url, name
    except Exception as e:
        print(f"Error fetching company information: {e}")
        return None, None, None

def stock_price(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    if start_date is not None:
        end_date = end_date.split('T')[0]
        df = yf.download(val, str(start_date), str(end_date))
        print("start date",start_date)
        print("end date",end_date)
    else:
        df = yf.download(val)
    print(df)    
    df.reset_index(inplace=True)
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
   # return dcc.Graph(figure=fig)
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(title='Candlestick Chart for ' + val,
                      xaxis_title='Date',
                      yaxis_title='Price')
    return dcc.Graph(figure=fig)

def indicators(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    if start_date is None:
        df_more = yf.download(val)
    else:
        end_date = end_date.split('T')[0]
        df_more = yf.download(val, str(start_date), str(end_date))
    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return dcc.Graph(figure=fig)

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

def forecast(n, n_days, val):
    if n is None or val is None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    # Extract and print the predicted data
    predicted_data = fig.data[0].y
    print("Predicted Close Prices for the next {} days:".format(n_days))
    for i, price in enumerate(predicted_data):
        print("Day {}: {:.2f}".format(i + 1, price))
    return dcc.Graph(figure=fig)

def prediction(stock, n_days):
    # Load the data
    df = yf.download(stock, period='3mo')
    df.reset_index(inplace=True)
    df['Day'] = df.index

    # Preprocess the data
    days = list()
    for i in range(len(df.Day)):
        days.append([i])
    X = days
    Y = df[['Close']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Train and select the model
    gsc = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid={'C': [0.001, 0.01, 0.1, 1, 100, 1000], 'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000], 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]}, cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)

    # Train the model
    best_svr.fit(x_train, y_train)
    
     # Make predictions on the test set
    y_pred = best_svr.predict(x_test)

    # Calculate accuracy metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(best_svr, X, Y.values.ravel(), cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

    # Predict and visualize the results
    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + x_test[-1][0]])
    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)
      
        

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=y_test.values.flatten(), mode='markers', name='data'))
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=best_svr.predict(x_test), mode='lines+markers', name='test'))
    fig = go.Figure()  # [Warning] Overwriting previous figure without reason

    fig.add_trace(go.Scatter(x=dates, y=best_svr.predict(output_days), mode='lines+markers', name='data'))
    fig.update_layout(title="Predicted Close Price of next " + str(n_days - 1) + " days", xaxis_title="Date", yaxis_title="Close Price")
    return fig
