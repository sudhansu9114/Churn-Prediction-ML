# Model is saved as model.pkl
# scaler is saved as scaler.pkl
# encoder is saved as encoder.pkl

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



model=pickle.load(open("model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
encoders=pickle.load(open("encoders.pkl","rb"))
Xtest = pickle.load(open("Xtest.pkl", "rb"))
Ytest = pickle.load(open("Ytest.pkl", "rb"))



st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")



st.title("📊 Customer Churn Prediction")

image = Image.open("4020769.jpg")  # Replace with your image file name
st.image(image, use_container_width=True)
st.markdown('<div class="subtitle">Enter customer details to check if they are likely to churn.</div>', unsafe_allow_html=True)

st.divider()

age=st.slider("🧓 enter Age",min_value=10,max_value=100,value=30)

gender=st.selectbox("⚧️ Enter Gender :",["Male","Female"])

tenure=st.number_input("📆 Enter Tenure (months) :",min_value=0,max_value=130,value=10)

monthly_charges=st.number_input("💵 Enter Monthly Charges :",min_value=20,max_value=150)

contract=st.selectbox("📝 Contract Type :",["Month-to-Month", "One-Year", "Two-Year"])

internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber Optic", "No"])

total_charges=st.number_input("💰 Enter Total Charges :",min_value=0,max_value=50000)

tech_support = st.selectbox("🛠️ Tech Support", ["Yes", "No"])

input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "ContractType": [contract],
    "InternetService": [internet],
    "TotalCharges": [total_charges],
    "TechSupport": [tech_support]
})


# Encoding.....
for col in ["Gender","ContractType","InternetService","TechSupport"]:
    input_df[col]=encoders[col].transform(input_df[col])
    
# Scaling......
scaled_input=scaler.transform(input_df)    

# predict......
if st.button("🚀 Predict Churn"):
    prediction=model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None
    
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn. (Confidence: {proba*100:.2f}%)")
        
        
    else:
        st.success(f"✅ The customer is likely to stay. (Confidence: {100 - proba*100:.2f}%)")
        
    # 📊 Evaluation Section
    st.divider()
    st.subheader("📈 Model Evaluation on Test Data")

    # Predict on test data
    y_test_pred = model.predict(Xtest)

  

    # 📊 Actual vs Predicted bar chart
    st.markdown("#### 📊 Actual vs Predicted Counts")
    actual_vs_pred = pd.DataFrame({'Actual': Ytest, 'Predicted': y_test_pred})
    st.bar_chart(actual_vs_pred.value_counts().unstack(fill_value=0))

    # 🔲 Confusion matrix heatmap
    st.markdown("#### 🔲 Confusion Matrix")
    cm = confusion_matrix(Ytest, y_test_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# 📝 Classification report
    st.markdown("#### 📄 Classification Report")
    report = classification_report(Ytest, y_test_pred,output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    

with st.sidebar:
    st.title("🧠 Model Info")
    
    st.markdown("""
    **Models Compared:**
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Machine (SVM)
    - Random Forest

    **Preprocessing:**
    - Label Encoding (Gender, Contract, etc.)
    - Feature Scaling (StandardScaler)

    **Model Selection:**
    - GridSearchCV with 5-fold CV
    - Best model saved with pickle

    **Prediction Target:**
    - `Churn` : Whether a customer is likely to leave

    **Accuracy Scores:**
    - Logistic Regression: 92.33%
    - KNN: 95.33%
    - SVM: 95.33%
    - Random Forest: 100.00%
    
    """)
    st.write("\n")
    st.write("\n")

    
    st.info("ℹ️ This app was built using Streamlit & Scikit-learn  (and a little bit of chatGPT here and there🫂🌟).")
    
    
    
    
    st.markdown("---")
    st.markdown("👨‍💻 **Built by:** [Sudhansu Sekhar Sahoo](mailto:sudhansusahoo9114@gmail.com)")
    st.markdown("📎 [GitHub](https://github.com/sudhansu9114)  |  💼 [LinkedIn](https://www.linkedin.com/in/sudhansu-sekhar-sahoo/)")