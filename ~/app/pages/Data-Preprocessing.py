import streamlit as st
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Gettting Data
df = st.file_uploader("Upload your file: ", type=['csv', 'xlsx', 'pickle'])
try:
  df = pd.read_csv(df)
  st.markdown("Your Data Record: ")
  st.dataframe(df)
except:
  st.write("Upload A CSV, EXCEL OR PICKLE FILE")

# Open Excel File
try:
  df = pd.read_excel(df, engine='openpyxl')
  st.markdown("Your Data Record: ")
  st.dataframe(df)
except:
  pass

# Read Pickle File
try:
  df = pd.read_pickle(df)
  st.markdown("Your Data Record: ")
  st.dataframe(df)
except:
  pass

try:
 
  X = st.multiselect("Choose X Values:",options=df.columns)
  y = st.selectbox("Choose Y Values:",options=df.columns)

  st.title('Preprocessing')
  cat_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None', 'Most frequent value'])
  num_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None', 'Median', 'Mean'])
  encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'])
  scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'])
  
  cat_cols_missing = st.selectbox("Choose cat_cols_missing value:",options=df.columns)
  num_cols_missing = st.selectbox("Choose num_cols_missing value:",options=df.columns)
  cat_cols = st.multiselect("Choose cat_cols value:",options=df.columns)
  num_cols = st.selectbox("Choose num_cols value:",options=df.columns)
  drop_cols = st.selectbox("Choose drop_cols value:",options=df.columns)
  
  def get_imputer(imputer):
    if imputer == 'None':
      return 'drop'
    if imputer == 'Most frequent value':
      return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    if imputer == 'Mean':
      return SimpleImputer(strategy='mean', missing_values=np.nan)
    if imputer == 'Median':
      return SimpleImputer(strategy='median', missing_values=np.nan)

  def get_encoder(encoder):
    if encoder == 'None':
      return 'drop'
    if encoder == 'OneHotEncoder':
      return OneHotEncoder(handle_unknown='ignore', sparse=False)
  def get_scaler(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()
    
  def get_pip_mis_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    pipeline = make_pipeline(get_imputer(imputer))
    pipeline.steps.append(('scaling', get_scaler(scaler)))
    return pipeline

  def get_pip_mis_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    pipeline = make_pipeline(get_imputer(imputer))
    pipeline.steps.append(('encoding', get_encoder(encoder)))
    return pipeline 
 
  def get_ml_algorithm(algorithm):
    if algorithm == 'Logistic regression':
        return LogisticRegression()
    if algorithm == 'Support vector':
        return SVC()
    if algorithm == 'K nearest neighbors':
        return KNeighborsClassifier()
    if algorithm == 'Random forest':
        return RandomForestClassifier()
        
  
  preprocessing = make_column_transformer( 
     (get_pip_mis_cat(cat_imputer_selected, encoder_selected) , cat_cols_missing),
     (get_pip_mis_num(num_imputer_selected, scaler_selected) , num_cols_missing),
     (get_encoder(encoder_selected), cat_cols),
     (get_scaler(scaler_selected), num_cols),
     ("drop" , drop_cols)
  )

  preprocessing_pipeline = Pipeline([
    ('preprocessing' , preprocessing)
  ])

  preprocessing_pipeline.fit(X)
  X_preprocessed = preprocessing_pipeline.transform(X)

  st.header('Preprocessed dataset')
  st.write(X_preprocessed)
  
  

  
except:
  pass
