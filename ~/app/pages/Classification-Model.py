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
      
target_selected = st.multiselect("Choose Column:",options=df.columns)
cat_cols_missing = st.multiselect("Choose Column:",options=df.columns)
num_cols_missing =  st.multiselect("Choose Column:",options=df.columns)
cat_cols =  st.multiselect("Choose Column:",options=df.columns)
num_cols =  st.multiselect("Choose Column:",options=df.columns)
drop_cols =  st.multiselect("Choose Column:",options=df.columns)

X = df.drop(columns = target_selected)
y = df[target_selected].values.ravel()

st.title('Model Selection')

classifier_list = ['Logistic regression', 'Support vector', 'K nearest neighbors', 'Random forest']
classifier_selected = st.sidebar.selectbox('', classifier_list)

pipeline = Pipeline([
    ('preprocessing' , preprocessing),
    ('ml', get_ml_algorithm(classifier_selected))
])

folds = KFold(n_splits = 10, shuffle=True, random_state = 0)
cv_score = cross_val_score(pipeline, X, y, cv=folds)
st.subheader('Results')
st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
st.write('Standard deviation : ', round(cv_score.std()*100,2), '%')
