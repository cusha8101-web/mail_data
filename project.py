import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mail_data = pd.read_csv(r"C:\Users\Dell\Desktop\Streamlit\datasets\mail_data.csv")

# mail_data['Category'] = mail_data['Category'].map({'ham': 0, 'spam': 1})
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1

mail_data['Category'] = mail_data['Category'].astype('int')

ham = mail_data[mail_data.Category == 0]
spam = mail_data[mail_data.Category == 1]
ham = ham.sample(n=1000)

x = mail_data['Message']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 2)
feature_extraction = TfidfVectorizer()

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
x_train_ex = feature_extraction.fit_transform(x_train)
x_test_ex = feature_extraction.transform(x_test)

print(x_train_ex)
print(x_test_ex)

model = LogisticRegression()

model.fit(x_train_ex, y_train)

x_train_ex_pred = model.predict(x_train_ex)
x_train_ex_accuracy = accuracy_score(x_train_ex_pred, y_train)

print(f"x_train accuracy score: {x_train_ex_accuracy}")

x_test_ex_pred = model.predict(x_test_ex)
x_test_ex_accuracy = accuracy_score(x_test_ex_pred, y_test)

#streamlit 
st.title("Email Spam Detection App")
input_mail = st.text_area("Enter the mail content below :")

if st.button("predict"):
  if input_mail.strip() !="":

    # Transform and Predict
    mail_transformed = feature_extraction.transform([input_mail])
    prediction = model.predict(mail_transformed)
    if prediction[0] == 1:
      st.error("this is a Spam mail")
    else:
      st.success("this is a ham(non-spam) mail")  

else:
    st.warning("Please enter some mail content to analyze")