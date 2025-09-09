# SMS Spam Classifier 📱✉️

A machine learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.  
This project demonstrates end-to-end implementation of **data preprocessing, feature engineering, model training, and evaluation**.

---

## 🚀 Project Overview
- Collected and cleaned SMS dataset.  
- Performed **Exploratory Data Analysis (EDA)** to understand message patterns and data imbalance.  
- Applied **TF-IDF vectorization** for text feature extraction.  
- Trained multiple models:
  - Naive Bayes  
  - Logistic Regression  
- Selected **Multinomial Naive Bayes** as the final model due to its superior performance.

---

## 📊 Results
- **Accuracy**: 98%  
- **Precision (Spam class)**: 100%  
- High recall and F1-score, demonstrating strong classification performance.  

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**: scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook  

---

## 📂 Project Structure
```
sms-spam-classifier/
│
├── notebooks/
│   └── SMS_Spam_Detection_Cleaned.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/NidhiKumari-Prog/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/SMS_Spam_Detection_Cleaned.ipynb
   ```

---

## 🎯 Key Learnings
- Hands-on experience with **text preprocessing** and **feature engineering** in NLP.  
- Applied multiple ML models and performed **model validation**.  
- Learned how to handle **imbalanced datasets** effectively.  

---

## 📌 Future Improvements
- Deploy the model as a web app using **Flask/Streamlit**.  
- Try advanced models like **SVM, Random Forest, or LSTMs**.  

---

## 👩‍💻 Author
**Nidhi Kumari**  
- MSc Mathematics and Scientific Computing @ NIT Warangal  
- [LinkedIn](https://www.linkedin.com/in/nidhi-kumari-nitw) | [GitHub](https://github.com/NidhiKumari-Prog)
