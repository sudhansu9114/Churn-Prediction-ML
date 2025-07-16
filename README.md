🧠 Customer Churn Prediction - Streamlit App  
A machine learning web app to predict whether a telecom customer is likely to churn. Built using Streamlit, scikit-learn, and pandas — with a touch of 🫂 ChatGPT.  
  
🚀 Features  
Interactive web app built with Streamlit  
User inputs for customer demographics and service info  
Trained model using best performance from multiple classifiers  
Live prediction with confidence percentage  
  
Evaluation visualizations:  
📄 Classification report  
📊 Actual vs Predicted bar chart  
🔲 Confusion matrix heatmap  
  
🧠 Model Info  
  
✅ Models Compared:  
Logistic Regression  
K-Nearest Neighbors  
Support Vector Machine (SVM)  
Random Forest  
  
🧪 Preprocessing:  
Label Encoding (Gender, Contract, etc.)  
Feature Scaling using StandardScaler  
  
🧬 Model Selection:  
GridSearchCV with 5-fold cross-validation  
Best model serialized using pickle  
  
🎯 Prediction Target:  
Churn — Whether a customer is likely to leave  
  
📈 Accuracy Scores:  
Model	Accuracy  
Logistic Regression	92.33%  
K-Nearest Neighbors	95.33%  
SVM	95.33%  
Random Forest	100.00% ✅  
  
🛠️ Tech Stack  
Python  
Pandas, NumPy  
Scikit-learn  
Matplotlib & Seaborn  
Streamlit  
  
💻 How to Run Locally  
git clone https://github.com/sudhansu9114/Churn-Prediction-ML.git  
cd churn-prediction-app  
pip install -r requirements.txt  
streamlit run app.py  
  
or  
  
Link : https://churn-prediction-sudhansu.streamlit.app  
  
👨‍💻 Built By  
Sudhansu Sekhar Sahoo  
  
🤖 Powered By  
Streamlit • scikit-learn • ChatGPT 🌟  
