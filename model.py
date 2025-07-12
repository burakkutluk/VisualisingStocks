import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from datetime import date, timedelta, datetime
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
import requests
import time
import os

# Financial Modeling Prep API Key - ücretsiz kayıt olun: https://financialmodelingprep.com/developer/docs
FMP_API_KEY = os.getenv('FMP_API_KEY', 'YOUR_FREE_API_KEY_HERE')  # Buraya API key'inizi koyun

def get_fmp_company_info(symbol):
    """Financial Modeling Prep API ile şirket bilgileri çekme"""
    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        params = {'apikey': FMP_API_KEY}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                company = data[0]
                return {
                    'name': company.get('companyName', 'N/A'),
                    'description': company.get('description', 'No description available'),
                    'logo': company.get('image', ''),
                    'website': company.get('website', ''),
                    'sector': company.get('sector', 'N/A'),
                    'industry': company.get('industry', 'N/A')
                }
        return None
    except Exception as e:
        print(f"FMP Company Info Error: {e}")
        return None

def get_fmp_historical_data(symbol, start_date=None, end_date=None):
    """Financial Modeling Prep API ile tarihsel veri çekme"""
    try:
        if start_date and end_date:
            # Belirli tarih aralığı için
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                'apikey': FMP_API_KEY,
                'from': start_date,
                'to': end_date
            }
        else:
            # Tüm veriler için
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {'apikey': FMP_API_KEY}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data and data['historical']:
                df = pd.DataFrame(data['historical'])
                # Sütun isimlerini yfinance formatına çevir
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                return df
        return None
    except Exception as e:
        print(f"FMP Historical Data Error: {e}")
        return None

def get_data_with_fallback(symbol, start_date=None, end_date=None):
    """FMP öncelikli, yfinance fallback veri çekme"""
    
    # Önce FMP API'sını dene
    print(f"FMP API ile {symbol} verisi çekiliyor...")
    fmp_data = get_fmp_historical_data(symbol, start_date, end_date)
    
    if fmp_data is not None and not fmp_data.empty:
        print("FMP API'sından veri başarıyla çekildi")
        return fmp_data, "FMP"
    
    # FMP başarısız olursa yfinance'ı dene
    print("FMP başarısız, yfinance deneniyor...")
    try:
        time.sleep(1)  # Rate limit için bekle
        if start_date and end_date:
            df = yf.download(symbol, start_date, end_date)
        else:
            df = yf.download(symbol, period="1y")
        
        if not df.empty:
            df.reset_index(inplace=True)
            print("yfinance'dan veri başarıyla çekildi")
            return df, "yfinance"
    except Exception as e:
        print(f"yfinance hatası: {e}")
    
    return None, "failed"

def update_data(n, val):
    if n is None or val is None:
        raise PreventUpdate
    
    # Önce FMP API'sını dene
    company_info = get_fmp_company_info(val)
    
    if company_info:
        description = f"{company_info['description']}\n\nSector: {company_info['sector']}\nIndustry: {company_info['industry']}"
        return description, company_info['logo'], company_info['name']
    
    # Fallback: yfinance
    try:
        print("FMP başarısız, yfinance company info deneniyor...")
        ticker = yf.Ticker(val)
        info = ticker.info
        name = info.get("longName", "Unknown Company")
        logo_url = info.get("logo_url", "")
        description = info.get("longBusinessSummary", "No description available")
        return description, logo_url, name
    except Exception as e:
        print(f"Error fetching company information: {e}")
        return "Company information not available", "", val.upper()

def stock_price(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    
    # Tarih formatını düzenle
    if start_date is not None and end_date is not None:
        if isinstance(end_date, str):
            end_date = end_date.split('T')[0]
        if isinstance(start_date, str):
            start_date = start_date.split('T')[0]
    
    # Veriyi çek
    df, source = get_data_with_fallback(val, start_date, end_date)
    
    if df is None or df.empty:
        # Hata durumunda boş grafik döndür
        fig = go.Figure()
        fig.add_annotation(
            text=f"Veri çekilemedi: {val}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=20)
        )
        fig.update_layout(title=f'Veri Bulunamadı - {val}')
        return dcc.Graph(figure=fig)
    
    print(f"Veri kaynağı: {source}")
    print(f"Veri boyutu: {len(df)} satır")
    
    # Candlestick grafik oluştur
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    
    fig.update_layout(
        title=f'Candlestick Chart for {val} (via {source})',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    
    return dcc.Graph(figure=fig)

def indicators(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    
    # Tarih formatını düzenle
    if start_date is not None and end_date is not None:
        if isinstance(end_date, str):
            end_date = end_date.split('T')[0]
        if isinstance(start_date, str):
            start_date = start_date.split('T')[0]
    
    # Veriyi çek
    df_more, source = get_data_with_fallback(val, start_date, end_date)
    
    if df_more is None or df_more.empty:
        # Hata durumunda boş grafik döndür
        fig = go.Figure()
        fig.add_annotation(
            text=f"İndikatör verisi çekilemedi: {val}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=20)
        )
        fig.update_layout(title=f'İndikatör Verisi Bulunamadı - {val}')
        return dcc.Graph(figure=fig)
    
    fig = get_more(df_more, source)
    return dcc.Graph(figure=fig)

def get_more(df, source=""):
    """EMA hesaplama ve görselleştirme"""
    try:
        # EMA hesapla
        df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EWA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Grafik oluştur
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))
        
        # 20-day EMA
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['EWA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='orange')
        ))
        
        # 50-day EMA (eğer yeterli veri varsa)
        if len(df) >= 50:
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['EWA_50'],
                mode='lines',
                name='EMA 50',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title=f'Price and Moving Averages (via {source})',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        print(f"EMA hesaplama hatası: {e}")
        # Fallback basit grafik
        fig = px.line(df, x="Date", y="Close", title="Close Price")
        return fig

def forecast(n, n_days, val):
    if n is None or val is None or n_days is None:
        raise PreventUpdate
    
    try:
        fig = prediction(val, int(n_days) + 1)
        # Extract and print the predicted data
        if fig and len(fig.data) > 0:
            predicted_data = fig.data[0].y
            print("Predicted Close Prices for the next {} days:".format(n_days))
            for i, price in enumerate(predicted_data):
                print("Day {}: {:.2f}".format(i + 1, price))
        return dcc.Graph(figure=fig)
    except Exception as e:
        print(f"Forecast error: {e}")
        # Hata grafik döndür
        fig = go.Figure()
        fig.add_annotation(
            text=f"Tahmin hesaplanamadı: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=f'Prediction Error - {val}')
        return dcc.Graph(figure=fig)

def prediction(stock, n_days):
    """ML ile fiyat tahmini - FMP API kullanarak"""
    try:
        # Son 3 aylık veriyi çek
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        df, source = get_data_with_fallback(stock, start_date, end_date)
        
        if df is None or df.empty or len(df) < 30:
            raise Exception("Yeterli veri bulunamadı (minimum 30 gün gerekli)")
        
        print(f"Tahmin için {len(df)} günlük veri kullanılıyor (kaynak: {source})")
        
        df.reset_index(drop=True, inplace=True)
        df['Day'] = df.index

        # Preprocess the data
        days = [[i] for i in range(len(df))]
        X = days
        Y = df[['Close']]
        
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        # Simplified parameter grid for faster execution
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1],
            'gamma': ['scale', 'auto', 0.001, 0.1]
        }

        # Train and select the model
        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'), 
            param_grid=param_grid, 
            cv=3,  # Reduced CV for speed
            scoring='neg_mean_absolute_error', 
            verbose=0, 
            n_jobs=-1
        )
        
        y_train_flat = y_train.values.ravel()
        grid_result = gsc.fit(x_train, y_train_flat)
        best_params = grid_result.best_params_
        
        best_svr = SVR(
            kernel='rbf', 
            C=best_params["C"], 
            epsilon=best_params["epsilon"], 
            gamma=best_params["gamma"]
        )

        # Train the model
        best_svr.fit(x_train, y_train_flat)
        
        # Make predictions on the test set
        y_pred = best_svr.predict(x_test)

        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"R-squared Score: {r2:.2f}")

        # Future predictions
        last_day = len(df) - 1
        output_days = [[last_day + i] for i in range(1, n_days)]
        
        # Generate future dates
        dates = []
        current = date.today()
        for i in range(n_days - 1):
            current += timedelta(days=1)
            dates.append(current)

        # Predict future prices
        future_predictions = best_svr.predict(output_days)

        # Create visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['Date'][-30:],  # Son 30 günü göster
            y=df['Close'][-30:],
            mode='lines+markers',
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=future_predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Predicted Close Price for Next {n_days-1} Days - {stock} (via {source})",
            xaxis_title="Date",
            yaxis_title="Close Price ($)",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise e
