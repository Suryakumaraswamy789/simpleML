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
occupation=st.sidebar.selectbox("JobRole",{'Tech-Support','Craft-repair','Other-service',"Sales","Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct","Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv","Protective-serv", "Armed-Forces"])
hours_per_week=st.sidebar.slider("Hours per weeek",1,86,40)
experience = st.sidebar.slider("Years of Experience:", 0, 40, 5)
# Build input DataFrame (must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours_per_week': [hours_per_week],
    'experience': [experience]
})
st.write("## Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f'Prediction: {prediction[0]}')

# Batch prediction
st.markdown("### Batch Prediction")
st.markdown("Upload a CSV file for batch prediction:")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.write(batch_data)
    
    batch_preds = model.predict(batch_data)
    batch_preds = [str(pred) for pred in batch_preds]
    st.write("Predictions:")
    st.write(batch_preds)
    # Batch predictions as downloadable CSV
     batch_preds_df = pd.DataFrame(batch_preds, columns=['Predicted Salary Class'])
    st.write(batch_preds_df)

    csv = batch_preds_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Predictions CSV', csv, file_name='predicted_classes.csv', mime='text/csv')
