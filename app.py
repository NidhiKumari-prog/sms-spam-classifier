import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import os


nltk_data_dir = os.path.expanduser('~/.nltk_data')
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora/stopwords')):
    nltk.download('stopwords', quiet=True)
if not os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt')):
    nltk.download('punkt', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = nltk.word_tokenize(text)
    y = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📩 SMS Spam Classifier")
st.write("Type a message below and find out whether it's spam or not!")

input_sms = st.text_area("Enter your message:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT a spam message!")
