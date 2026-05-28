import joblib
import pandas as pd
import streamlit as st


#page setup
st.set_page_config(page_icon="💓",page_title="Heart Disease Predictor",layout="wide")

with st.sidebar:
    st.title("Heart Disease Predictor")
    # st.image("heart.jpg")


#load the dataset
df = pd.read_csv("cleaned_data.csv")
#load the model
model = joblib.load("HD_model.joblib")

# Column names matching the training data
FEATURE_COLUMNS = ["age","sex","cp","trestbps","chol","fbs",
                   "restecg","thalach","exang","oldpeak","slope","ca","thal"]

#user input
with st.container(border=True):
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age: ",min_value=1,max_value=100,step=5)
        sex = st.radio("Gender: ",options=["Male","Female"],horizontal=True)
        sex = 1 if sex=="Male" else 0
        d = {"Typical angina":0,"Atypical angina":1,"non-anginal pain":2,"Asymptomatic":3}
        cp = st.selectbox("Chest pain type: ",options=d.keys())
        cp = d[cp]
        trestbps =st.number_input("Resting BP: ",min_value=50,max_value=250,step=5)
        chol =st.number_input("Cholestrol: ",min_value=50,max_value=600,step=5)
        fbs = st.radio("Fasting Blood Sugar: ",options=["Yes","No"],horizontal=True)
        fbs  =1 if fbs=="Yes" else 0

    with col2:
        d = {"Normal":0,"Having ST-T wave abnormality":1,"Left ventricular hypertrophy":2}
        restecg = st.selectbox("Resting ECG: ",options=d.keys())
        restecg = d[restecg]
        thalach  =st.number_input("Max Heart Rate: ",min_value=50,max_value=250,step=5)
        exang =st.radio("Exer induced angina: ",options=["Yes","No"],horizontal=True)
        exang  =1 if exang=="Yes" else 0
        oldpeak  =st.number_input("Oldpeak ",min_value=0.0,max_value=10.0,step=1.0)
        d = {"upsloping":0,"flat":1,"downsloping":2}
        slope = st.selectbox("Slope: ",options=d.keys())
        slope = d[slope]
        ca = st.selectbox("Num of Major Vessels: ",options=[0,1,2,3,4])
        d = { "normal": 1 , "fixed defect" :2 , "reversable defect":3}
        thal = st.selectbox("Thal:  ",options=d.keys())
        thal = d[thal]

    if st.button("Predict"):
        # ✅ Use a named DataFrame so feature order matches training data
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURE_COLUMNS
        )
        pred = model.predict(input_data)[0]

        if pred == 0:
            st.subheader("Low Risk of Heart Disease")
            # st.image("lowrisk.png",width=150)
        else:
            st.subheader("High Risk of Heart Disease")
            # st.image("highrisk.png",width=150)