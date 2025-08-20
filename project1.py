import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
heart_data = pd.read_csv(r'C:\Users\Dell\Desktop\Streamlit\datasets\heart_disease_data.csv')

# Load dataset

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

st.title("Heart Disease Predictor")
st.write("fill in the deatils below to check if there is heart disease or not.")

age = st.number_input('age',min_value=1,max_value=120, value=30)
sex = st.selectbox('sex', (1,0))
cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=0)
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
restecg = st.number_input('Resting Electrocardiographic results (0-2)', min_value=0, max_value=2, value=1)
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', (0, 1))
oldpeak = st.number_input('ST depression induced by exercise', 0.0, 10.0, 1.0, step=0.1)
slope = st.number_input('Slope of the peak exercise ST segment (0-2)', 0, 2, 1)
ca = st.number_input('Number of major vessels colored by fluoroscopy (0-4)', 0, 4, 0)
thal = st.number_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 0, 2, 1)
target = st.number_input('target (0 = normal; 1 = detected)', 0 , 1)



# Predict
if st.button("Predict"):
    # input_data = np.array(62,0,0,140,268,0,0,160,0,3.6,0,2,2)
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])
        
    input_data_reshaped = input_data.reshape(1,-1)

    # Get the feature names from the training data
    feature_names = X_train.columns

    # Convert the reshaped input data to a pandas DataFrame with feature names
    input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)
    prediction = model.predict(input_data_df)


    if (prediction[0]== 0):
        st.success('The Person does not have a Heart Disease')
    else:
        st.error('The Person has Heart Disease')