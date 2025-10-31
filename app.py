import streamlit as st
import pandas as pd
import time
from datetime import datetime

import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import plotly.graph_objects as go

import torch
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

st.set_page_config(
      page_title="Sales Predictor-AI Project",
      page_icon="📈",
      layout="wide",
      initial_sidebar_state="expanded",
)

# Preprocessing
@st.cache_data
def merge(B, C, A):
  i = j = k = 0

  # Convert 'Date' columns to datetime.date objects
  B['Date'] = pd.to_datetime(B['Date']).dt.date
  C['Date'] = pd.to_datetime(C['Date']).dt.date
  A['Date'] = pd.to_datetime(A['Date']).dt.date

  while i < len(B) and j < len(C):
    if B['Date'].iloc[i] <= C['Date'].iloc[j]:
      A['Date'].iloc[k] = B['Date'].iloc[i]
      A['Sales'].iloc[k] = B['Sales'].iloc[i]
      i += 1
      
    else:
      A['Date'].iloc[k] = C['Date'].iloc[j]
      A['Sales'].iloc[k] = C['Sales'].iloc[j]
      j += 1
    k += 1

  while i < len(B):
    A['Date'].iloc[k] = B['Date'].iloc[i]
    A['Sales'].iloc[k] = B['Sales'].iloc[i]
    i += 1
    k += 1

  while j < len(C):
    A['Date'].iloc[k] = C['Date'].iloc[j]
    A['Sales'].iloc[k] = C['Sales'].iloc[j]
    j += 1
    k += 1

  return A

@st.cache_data
def merge_sort(dataframe):
  if len(dataframe) > 1:
      center = len(dataframe) // 2
      left = dataframe.iloc[:center]
      right = dataframe.iloc[center:]
      merge_sort(left)
      merge_sort(right)

      return merge(left, right, dataframe)

  else:
      return dataframe

@st.cache_data
def drop (dataframe):
  def get_columns_containing(dataframe, substrings):
    return [col for col in dataframe.columns if any(substring.lower() in col.lower() for substring in substrings)]

  columns_to_keep = get_columns_containing(dataframe, ["date", "sale"])
  dataframe = dataframe.drop(columns=dataframe.columns.difference(columns_to_keep))
  dataframe = dataframe.dropna()
    
  return dataframe

@st.cache_data
def date_format(dataframe):
  for i, d, s in dataframe.itertuples():
    dataframe['Date'][i] = dataframe['Date'][i].strip()

  for i, d, s in dataframe.itertuples():
    new_date = datetime.strptime(dataframe['Date'][i], "%m/%d/%Y").date()
    dataframe['Date'][i] = new_date

  return dataframe

@st.cache_data
def group_to_three(dataframe):
  dataframe['Date'] = pd.to_datetime(dataframe['Date'])
  dataframe = dataframe.groupby([pd.Grouper(key='Date', freq='3D')])['Sales'].mean().round(2)
  dataframe = dataframe.replace(0, np.nan).dropna()

  return dataframe

@st.cache_data
def series_to_df_exogenous(series):
  dataframe = series.to_frame()
  dataframe = dataframe.reset_index()
  dataframe = dataframe.set_index('Date')
  dataframe = dataframe.dropna()
  # Create the eXogenous values
  dataframe['Sales First Difference'] = dataframe['Sales'] - dataframe['Sales'].shift(1)
  dataframe['Seasonal First Difference'] = dataframe['Sales'] - dataframe['Sales'].shift(12)
  dataframe = dataframe.dropna()
  return dataframe

@st.cache_data
def dates_df(dataframe):
  dataframe = dataframe.reset_index()
  dataframe['Date'] = dataframe['Date'].dt.strftime('%B %d, %Y')
  dataframe[dataframe.columns] = dataframe[dataframe.columns].astype(str)
  return dataframe

@st.cache_data
def get_forecast_period(period):
  return round(period / 3)

# SARIMAX Model
@st.cache_data
def train_test(dataframe, n):
  training_y = dataframe.iloc[:-n,0]
  test_y = dataframe.iloc[-n:,0]
  test_y_series = pd.Series(test_y, index=dataframe.iloc[-n:, 0].index)
  training_X = dataframe.iloc[:-n,1:]
  test_X = dataframe.iloc[-n:,1:]
  future_X = dataframe.iloc[0:,1:]
  return (training_y, test_y, test_y_series, training_X, test_X, future_X)

@st.cache_data
def test_fitting(dataframe, Exo, trainY):
  trainTestModel = auto_arima(X = Exo, y = trainY, start_p=1, start_q=1,
                         test='adf',min_p=1,min_q=1,
                         max_p=3, max_q=3, m=12,
                         start_P=2, start_Q=2, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True, maxiter = 50)
  model = trainTestModel
  return model
  
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)).round(4)  # MAPE
    rmse = (np.mean((forecast - actual)**2)**.5).round(2)  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                            actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                            actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'rmse':rmse, 'corr':corr, 'min-max':minmax})

@st.cache_data
def sales_growth(dataframe, fittedValues):
    sales_growth = fittedValues.to_frame()
    sales_growth = sales_growth.reset_index()
    sales_growth.columns = ("Date", "Sales")
    sales_growth = sales_growth.set_index('Date')

    sales_growth['Sales'] = (sales_growth['Sales']).round(2)

    # Calculate and create the column for sales difference and growth
    sales_growth['Forecasted Sales First Difference']=(sales_growth['Sales']-sales_growth['Sales'].shift(1)).round(2)
    sales_growth['Forecasted Sales Growth']=(((sales_growth['Sales']-sales_growth['Sales'].shift(1))/sales_growth['Sales'].shift(1))*100).round(2)

    # Calculate and create the first row for sales difference and growth
    sales_growth['Forecasted Sales First Difference'].iloc[0] = (dataframe['Sales'].iloc[-1]-dataframe['Sales'].iloc[-2]).round(2)
    sales_growth['Forecasted Sales Growth'].iloc[0]=(((dataframe['Sales'].iloc[-1]-dataframe['Sales'].iloc[-2])/dataframe['Sales'].iloc[-1])*100).round(2)

    return sales_growth

@st.cache_data
def merge_forecast_data(actual, predicted, future): # debug
    actual = actual.to_frame()
    print("BEFORE RENAME ACTUAL")
    print(actual)
    actual.rename(columns={actual.columns[0]: "Actual Sales"}, inplace=True)
    print("ACTUAL")
    print(actual)

    predicted = predicted.to_frame()
    predicted.rename(columns={predicted.columns[0]: "Predicted Sales"}, inplace=True)
    print("PREDICTED")
    print(predicted)

    future = future.to_frame()
    future = future.rename_axis('Date')
    future.rename(columns={future.columns[0]: "Forecasted Future Sales"}, inplace=True)
    print("FUTURE")
    print(future)

    merged_dataframe = pd.concat([actual, predicted, future], axis=1)
    print("MERGED DATAFRAME")
    print(merged_dataframe)
    merged_dataframe = merged_dataframe.reset_index()
    print("MERGED DATAFRAME RESET INDEX")
    print(merged_dataframe)
    return merged_dataframe

def interpret_mape(mape_score):
  score = (mape_score * 100).round(2)
  if score < 10:
    interpretation = "Great"
    color = "green"
  elif score < 20:
    interpretation = "Good"
    color = "seagreen"
  elif score < 50:
    interpretation = "Relatively good"
    color = "orange"
  else:
    interpretation = "Poor"
    color = "red"
  return score, interpretation, color

# TAPAS Model

@st.cache_resource
def load_tapas_model():
  model_name = "google/tapas-large-finetuned-wtq"
  tokenizer = TapasTokenizer.from_pretrained(model_name)
  model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
  pipe = pipeline("table-question-answering", model=model, tokenizer=tokenizer)
  return pipe

pipe = load_tapas_model()

def get_answer(table, query):
    answers = pipe(table=table, query=query)
    return answers

def convert_answer(answer):
    if answer['aggregator'] == 'SUM':
      cells = answer['cells']
      converted = sum(float(value.replace(',', '')) for value in cells)
      return converted

    if answer['aggregator'] == 'AVERAGE':
      cells = answer['cells']
      values = [float(value.replace(',', '')) for value in cells]
      converted = sum(values) / len(values)
      return converted

    if answer['aggregator'] == 'COUNT':
      cells = answer['cells']
      converted = sum(int(value.replace(',', '')) for value in cells)
      return converted

    else:

      return answer['answer']

def get_converted_answer(table, query):
    converted_answer = convert_answer(get_answer(table, query))
    return converted_answer

# Session States
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False

if 'forecasted' not in st.session_state:
    st.session_state.forecasted = False

# Web Application
st.title("Forecasting Dashboard 📈")
if not st.session_state.uploaded:
  st.subheader("Welcome User, get started forecasting by uploading your file in the sidebar!")

# Sidebar Menu
with st.sidebar:
    st.title("Forecaster v1.1")
    st.subheader("An intelligent sales forecasting system")
    uploaded_file = st.file_uploader("Upload your store data here to proceed (must atleast contain Date and Sales)", type=["csv"])
    if uploaded_file is not None:
      date_found = False
      sales_found = False
      df = pd.read_csv(uploaded_file, parse_dates=True)
      for column in df.columns:
        if 'Date' in column:  
          date_found = True
        if 'Sales' in column:
          sales_found = True
      if(date_found == False or sales_found == False):
        st.error('Please upload a csv containing both Date and Sales...')
        st.stop()

      st.success("File uploaded successfully!")
      st.write("Your uploaded data:")
      st.write(df)

      df = drop(df)
      df = date_format(df)
      merge_sort(df)
      series = group_to_three(df)

      st.session_state.uploaded = True

    with open('sample.csv', 'rb') as f:
       st.download_button("Download our sample CSV", f, file_name='sample.csv')

if (st.session_state.uploaded):
  st.subheader("Sales History")
  st.line_chart(series)
  
  MIN_DAYS = 30
  MAX_DAYS = 90
  period = st.slider('How many days would you like to forecast?', min_value=MIN_DAYS, max_value=MAX_DAYS)
  forecast_period = get_forecast_period(period)

  forecast_button = st.button(
    'Start Forecasting',
    key='forecast_button',
    type="primary",
  )

  if (forecast_button or st.session_state.forecasted):
    df = series_to_df_exogenous(series)
    n_periods = round(len(df) * 0.2)
    print(n_periods) # debug

    train = train_test(df, n_periods)
    training_y, test_y, test_y_series, training_X, test_X, future_X = train
    train_test_model = test_fitting(df, training_X, training_y)
    
    print(df) # debug
    print(len(df)) # debug

    future_n_periods = forecast_period
    fitted, confint = train_test_model.predict(X=test_X, n_periods=n_periods, return_conf_int=True)
    index_of_fc = test_y_series.index

    # make series for plotting purpose
    fitted_series = pd.Series(fitted)
    fitted_series.index = index_of_fc
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    #Future predictions
    frequency = '3D'
    future_fitted, confint = train_test_model.predict(X=df.iloc[-future_n_periods:,1:], n_periods=future_n_periods, return_conf_int=True, freq=frequency)
    future_index_of_fc = pd.date_range(df['Sales'].index[-1], periods = future_n_periods, freq=frequency)

    # make series for future plotting purpose
    future_fitted_series = pd.Series(future_fitted)
    future_fitted_series.index = future_index_of_fc
    # future_lower_series = pd.Series(confint[:, 0], index=future_index_of_fc)
    # future_upper_series = pd.Series(confint[:, 1], index=future_index_of_fc)

    future_sales_growth = sales_growth(df, future_fitted_series)

    test_y, predictions = np.array(test_y), np.array(fitted)
    print("Test Y:", test_y) # debug
    print("Prediction:", fitted) # debug
    score = forecast_accuracy(predictions, test_y)
    print("Score:", score) # debug
    mape, interpretation, mape_color = interpret_mape(score['mape'])
    
    print(df)
    print(df['Sales'])
    merged_data = merge_forecast_data(df['Sales'], fitted_series, future_fitted_series)

    col_charts = st.columns(2)

    print(merged_data) # debug
    print(merged_data.info)
    print(merged_data.dtypes)
    with col_charts[0]:
      fig_compare = go.Figure()
      fig_compare.add_trace(go.Scatter(x=merged_data[merged_data.columns[0]], y=merged_data['Actual Sales'], mode='lines', name='Actual Sales'))
      fig_compare.add_trace(go.Scatter(x=merged_data[merged_data.columns[0]], y=merged_data['Predicted Sales'], mode='lines', name='Predicted Sales', line=dict(color='#006400')))
      fig_compare.update_layout(title='Historical Sales Data', xaxis_title='Date', yaxis_title='Sales')
      st.plotly_chart(fig_compare, use_container_width=True)

    with col_charts[1]:
      fig_forecast = go.Figure()
      fig_forecast.add_trace(go.Scatter(x=merged_data[merged_data.columns[0]], y=merged_data['Actual Sales'], mode='lines', name='Actual Sales'))
      fig_forecast.add_trace(go.Scatter(x=merged_data[merged_data.columns[0]], y=merged_data['Forecasted Future Sales'], mode='lines', name='Future Forecasted Sales', line=dict(color=mape_color)))
      fig_forecast.update_layout(title='Forecasted Sales Data', xaxis_title='Date', yaxis_title='Sales')
      st.plotly_chart(fig_forecast, use_container_width=True)
      st.write(f"MAPE score: {mape}% - {interpretation}")

    df = dates_df(future_sales_growth)

    col_table = st.columns(2)
    with col_table[0]:
      col_table[0].subheader(f"Forecasted sales in the next {period} days")
      col_table[0].write(df)

    with col_table[1]:
      col_table[1] = st.subheader("Question-Answering")
      with st.form("question_form"):
        question = st.text_input('Ask a Question about the Forecasted Data', placeholder="What is the total sales in the month of December?")
        query_button = st.form_submit_button(label='Generate Answer')
      if query_button or question:
          answer = get_converted_answer(df, question)
          if answer is not None:
            st.write("The answer is:", answer)
          else:
            st.write("Answer is not found in table")
    st.session_state.forecasted = True


# Hide Streamlit default style
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)