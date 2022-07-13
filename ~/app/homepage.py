import streamlit as st
import pandas as pd
import numpy as np
import os 
import datetime
import time
from PIL import Image 


# App Header
col1, col2 = st.columns(2)
image = Image.open('image.png')

col1.markdown('''# **Machine Learning Exploratory Tool**
A Machine Learning Classification Web Application.
''')
col1.write("This Web Application predicts with machine classification algorithmn ")
col2.image(image)

