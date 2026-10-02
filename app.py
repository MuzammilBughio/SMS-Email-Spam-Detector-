import streamlit as st  
import pickle 
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Data Preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):

  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for word in text:
    if word.isalnum():
      y.append(word)  

  text = y[:]
  y.clear()

  for word in text:
    if word not in stopwords.words('english') and word not in string.punctuation:
      y.append(word)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl' , 'rb'))
model = pickle.load(open('model.pkl' , 'rb'))

st.title("SMS/Email Spam Classifier")

input_sms = st.text_area(
    label="📩 Enter SMS message",
    placeholder="Type or paste your SMS here...",
    height=180,
    max_chars=500,
    help="We'll analyze the message and predict whether it is Spam or Not Spam.",
    label_visibility="visible",
    width="stretch"
)
if st.button('Predict'):

    # Preprocess
    transform_sms = transform_text(input_sms)
    
    # Vectorize
    vector_input = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")