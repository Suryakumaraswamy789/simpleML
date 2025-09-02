import streamlit as st
import pandas as pd
import joblib
model=joblib.load("chat.pkl")
st.set_page_config(page_title="Employee Salary Prediction",page_icon="",layout="centered")
st.title("employee Salary Prediction App")
st.markdown("predict wheatheran employee earn >50k or <50k based on input feature")
st.sidebar.header("Input Employee Details")
age=st.sidebar.selectbox("Age",18,58,30)
education=st.sidebar.selectbox("Education Level",['Bachelor','Masters','Phd','HS-grad','Assoc','some-college'])
occupation=st.sidebar.selectbox("JobRole",['Tech-Support','Craft-repair','Other-service',"Sales","Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct","Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv","Protective-serv", "Armed-Forces"])
hours_per_week=st.sidebar.slider("Hours per weeek",1,86,40)
experience = st.sidebar.slider("Years of Experience:", 0, 40, 5)
# Build input DataFrame (must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours_per_week': [hours_per_week],
    'experience': [experience]})
st.write("## Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f'Prediction: {prediction[0]}')
