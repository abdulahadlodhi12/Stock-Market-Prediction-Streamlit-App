# Import the libararires for Stock Market Prediction App
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from keras.models import Sequential
from keras.layers import LSTM , Dense
from sklearn.preprocessing import MinMaxScaler



# Set the Window Size of our Stock Market App
st.set_page_config(layout = 'wide' , initial_sidebar_state='collapsed')

# Set the title and subheader of the App
st.title("Stock Market Prediction App")
st.subheader("Stock Market Prediction on yFinance Data")

# Creating a Sidebar and taking the input of the start_date and the End Date from the user

# Sidebar Area

# Start and end date
start_date = st.sidebar.date_input("Enter The Start Date",date(2024,1,1))
end_date = st.sidebar.date_input("Enter The End Date",date(2025,12,31))

# Stocks Reprentation or ticker names and sidebar implementation using selectbox
stocks_ticker = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "NVDA", "BRK-B", "JNJ",
    "V", "JPM", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "CRM",
    "INTC", "CSCO", "PEP", "KO", "MCD", "NFLX", "T", "VZ", "BA", "XOM"
]

ticker_side_bar_selectbox = st.sidebar.selectbox("Select Any Stock For Prediciton",stocks_ticker)






# Now downloading the data from the yfinance library of the selected ticker
data = yf.download(ticker_side_bar_selectbox , start=start_date, end=end_date)
if data.empty:
    st.error(f"No data found for {ticker_side_bar_selectbox} from {start_date} to {end_date}.")
    st.stop() 
# Add data as the index
data.insert(0,"Date",data.index,True)
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
data.reset_index(drop=True,inplace=True)
st.write(f"Data Collection Starts From {start_date} To {end_date}")
st.write(data)


# PLot the data Visually
st.header("Plotting The Data Visually")
fig = px.line(data,x='Date_',y=data.columns,title='Closing Price of the Stock',width = 1500,height =800)
st.plotly_chart(fig)



# Selecting columns from user for forecasting the results of the stock market
forecasting_cols = st.sidebar.selectbox("Select The Columns You Want To Apply Forecasting" , data.columns[1:])

# Subsetting the data for future forecasting
data = data[["Date_" , forecasting_cols]]
st.subheader("Selected Data For Prediction Will be:")
st.write(data)



# Model Selection and selectbox Implementation
model_name = ['SARIMA','Random Forest','LSTM']
selected_model = st.sidebar.selectbox("Select The Model Name From Which You Have To Take Out Prediction",model_name)

if selected_model == 'SARIMA':

    # Apply adfuller to test the stationarity of the data
    st.header("Is Data Stationary?")
    adf_result = adfuller(data[forecasting_cols])[1] < 0.05
    st.write("Stationarity of the data:", adf_result)
    if not adf_result:
        st.write("This Means Data Is Not Stationary")
    else:
        st.write("This Means Data Is Stationary")

    # Decompose the data
    st.subheader("Checking The Seasonality Of The Data")
    decompose = seasonal_decompose(data[forecasting_cols], model='additive', period=12)
    st.write(decompose.plot())
    st.plotly_chart(px.line(data, x='Date_', y=decompose.trend, title="Trend", width=1500, height=800, labels={'x':'Date','y':'Price'}).update_traces(line_color="Blue"))
    st.plotly_chart(px.line(data, x='Date_', y=decompose.seasonal, title="Seasonality", width=1500, height=800, labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
    st.plotly_chart(px.line(data, x='Date_', y=decompose.resid, title="Residuals", width=1500, height=800, labels={'x':'Date','y':'Price'}).update_traces(line_color='Red', line_dash="dot"))

    # SARIMA parameters
    p = st.slider("Enter The P Value For SARIMA", 0, 5, 2)
    d = st.slider("Enter The D Value For SARIMA", 0, 5, 1)
    q = st.slider("Enter The Q Value For SARIMA", 0, 5, 2)
    seasonal_order = st.number_input("Select The Value Of Seasonality:", 0, 24, 12)

    # Fit SARIMA model
    model = sm.tsa.statespace.SARIMAX(data[forecasting_cols], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
    model = model.fit()
    st.subheader("Summary Of The Model")
    st.write(model.summary())
    st.write('---')

    # In-sample predictions and metrics

    in_sample_pred = model.get_prediction(start=0, end=len(data)-1)
    in_sample_mean = in_sample_pred.predicted_mean

    mse = mean_squared_error(data[forecasting_cols], in_sample_mean)
    rmse = np.sqrt(mse)
    r2 = r2_score(data[forecasting_cols], in_sample_mean)

    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"R² Score: {r2:.4f}")
    st.write('---')

    # Forecasting future values
    st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>", unsafe_allow_html=True)
    forecast_period = st.number_input("Select How Many Days You Want To Predict", 1, 365, 10)

    predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
    predicted_mean = predictions.predicted_mean

    # Create proper Date index for predictions
    pred_dates = pd.date_range(start=data['Date_'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')
    predictions_df = pd.DataFrame({'Date': pred_dates, 'Predicted': predicted_mean.values})

    # Ensure actual data has Date column
    data_plot = data.copy()
    if 'Date' not in data_plot.columns:
        data_plot['Date'] = data_plot['Date_']

    # Plot actual vs predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_plot["Date"], y=data_plot[forecasting_cols], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions_df["Date"], y=predictions_df["Predicted"], mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)



elif selected_model == 'Random Forest':
    # Splitting the data and applying random forest algorithm to the data
    st.header("Random Forest Regressor")

    # Spiliting the data into training and testing dataset
    train_size = int(len(data) * 0.8)
    train_data,test_data = data[:train_size] , data[train_size:]

    # Splitting the data into x_train,y_train,x_test,y_test
    x_train,y_train = train_data['Date_'],train_data[forecasting_cols]
    x_test,y_test = test_data['Date_'],test_data[forecasting_cols]

    # Call the model and fit or train the model
    rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
    rf_model.fit(x_train.values.reshape(-1,1),y_train.values)

    # Predict the model
    model_pred = rf_model.predict(x_test.values.reshape(-1,1))
    mse = mean_squared_error(y_test,model_pred)
    r2 = r2_score(y_test,model_pred)
    rmse = np.sqrt(mse)
    st.write(f'Root Mean Squared Error: {rmse}')
    st.write(f'R2 Score Of The Predictions: {r2}')

    # Combine the train and testing data for plotting the data
    concat_dataset = pd.concat([train_data,test_data])

    # Now Plot the graph of the data
    fig = go.Figure()
    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=concat_dataset["Date_"], y=concat_dataset[forecasting_cols], mode='lines', name='Actual', line=dict(color='blue')))
    # Add predicted data to the plot
    fig.add_trace(
        go.Scatter(x=test_data["Date_"], y=model_pred, mode='lines', name='Predicted',
               line=dict(color='red')))
    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    # Display the plot
    st.plotly_chart(fig)


elif selected_model == 'LSTM':
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[forecasting_cols].values.reshape(-1,1))

    # Train/test split
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Sequence creation function
    def create_sequence(dataset , seq_length):
        X,y = [],[]
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length , 0])
            y.append(dataset[i+seq_length,0])
        return np.array(X), np.array(y)

    seq_length = st.slider("Select The Sequence Length: " , 1,50,10)
    st.write("<p style='color:#FFFFFF; font-size: 15px;'>Sequence Means How Many Days Your Are Giving To Model In Order To Predict The Future Price</p>",unsafe_allow_html=True)
    x_train, y_train = create_sequence(train_data,seq_length)
    x_test, y_test   = create_sequence(test_data,seq_length)

    # Reshape for LSTM [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test  = np.reshape(x_test,  (x_test.shape[0],  x_test.shape[1],  1))

    # Build the LSTM model
    lstm = Sequential()
    lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    lstm.add(LSTM(units=50))
    lstm.add(Dense(units=1))

    # Compile and train LSTM
    lstm.compile(optimizer="Adam",loss='mean_squared_error',metrics=["mae", "mape"])
    lstm.fit(x_train, y_train, epochs=100, batch_size=32,verbose=1)

    # Predictions on train + test
    pred_train = lstm.predict(x_train)
    pred_test  = lstm.predict(x_test)

    # Inverse scale
    pred_train = scaler.inverse_transform(pred_train).ravel()
    pred_test  = scaler.inverse_transform(pred_test).ravel()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
    y_test_actual  = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()

    # Metrics
    st.write(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_actual, pred_train))}")
    st.write(f"Train R2_Score:   {r2_score(y_train_actual, pred_train)}")
    st.write(f"Test  RMSE: {np.sqrt(mean_squared_error(y_test_actual, pred_test))}")
    st.write(f"Test  R2_Score:   {r2_score(y_test_actual, pred_test)}")

    # ---- FUTURE FORECASTING ----
    future_days = st.slider("How many future days to predict?", 1, 120, 30)

    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    future_predictions = []

    for _ in range(future_days):
        next_pred = lstm.predict(last_seq)[0][0]
        future_predictions.append(next_pred)

        # Slide window
        last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

    # Inverse scale future predictions
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).reshape(-1)

    # Future dates
    last_date = data["Date_"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # ---- PLOT ----
    fig = go.Figure()

    # Actual historical data
    fig.add_trace(go.Scatter(
        x=data["Date_"],
        y=data[forecasting_cols],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Future prediction line (AFTER actual ends)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Future Predicted',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Actual vs Future Predicted (LSTM)',
        xaxis_title='Date',
        yaxis_title='Price',
        width=1200,
        height=450
    )

    st.plotly_chart(fig)
else:
    st.warning("Enter Or Select The Given Model For Prediction")


st.write("Model selected:", selected_model)

# urls of the images
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
# redirect urls
github_redirect_url = "https://github.com/abdulahadlodhi12"
linkedIN_redirect_url = 'https://www.linkedin.com/in/abdul-ahad-lodhi-94b36b283/'
kaggle_redirect_url = 'https://www.kaggle.com/gamerahad/code'

# adding a footer
st.markdown("""
<style>
            
html, body, .main {
    height: 100%;
}

.footer {
    width: 100%;
    background-color: #2A2D36;
    color: #DDE6F2;
    text-align: center;
    padding:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">Made with ❤️ by Abdul Ahad Lodhi <a href="{github_redirect_url}">Github  </a>'
             f'<a href="{linkedIN_redirect_url}">LinkedIn  </a>'
             f'<a href="{kaggle_redirect_url}">Kaggle</a> | Credits: Dr.Ammaar Tufail | Muhammad Bilal Butt</div>', unsafe_allow_html=True)










    











