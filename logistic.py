import os
import streamlit as st
import streamlit.components.v1 as com
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.chdir('/Users/apple/Downloads/logistic_regression')
data=pd.read_csv('2.csv',na_values=["??","####"])
x=data.iloc[:,[0,1]].values
y=data.iloc[:,-1].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
s=StandardScaler()
xtrain=s.fit_transform(xtrain)
xtest=s.transform(xtest)
model=LogisticRegression(random_state=0)
model.fit(xtrain,ytrain)

st.title(":red[Machine Learning Web-App]")
st.header("Check your eligibility for a computer purchase!!!(By Logistic Regression)")
age=st.slider('How old are you?',
                    min_value=0,
                    max_value=100,
                    step=1)
salary=st.number_input('Enter your salary',
                    min_value=0,
                    max_value=100000000,
                    step=1)

result=model.predict(s.transform([[age,salary]]))[0]
if(st.button("Result")):
    if(result==1):
        st.header("Resuls: ")
        st.write('You are eligible for buying a laptop :computer:')
    else:
        st.header('Results: ')
        st.write('You are not eligible for buying a laptop :computer:')

