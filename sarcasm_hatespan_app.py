
import pandas as pd
import numpy as np
import seaborn as sns
import re
import string
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# Download stopwords and WordNet
nltk.download('stopwords')
nltk.download('wordnet')
stopword = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
unique_sarcastic_words=['little', 'new', 'employee', 'job', 'human', 'report', 'self', 'tell', 'local', 'good', 'every', 'announces', 'study', 'dad', 'couple', 'get', 'parent', 'god', 'never', 'last', 'come', 'woman', 'day', 'million', 'doesnt', 'teen', 'entire', 'party', 'yearold', 'see', 'getting', 'friend', 'one', 'he', 'man', 'find', 'cant', 'office', 'hour', 'bush', 'area', 'around', 'enough', 'guy', 'minute', 'work', 'introduces', 'american', 'going', 'nation']
unique_hate_speech_words=['life', 'yall', 'cunt', 'fucking', 'ya', 'bitch', 'lmao', 'bad', 'gonna', 'fuck', 'get', 'come', 'wit', 'damn', 'lil', 'dat', 'like', 'faggot', 'aint', 'cause', 'hoe', 'money', 'always', 'stop', 'yo', 'rt', 'eat', 'real', 'little', 'ill', 'nigga', 'tell', 'u', 'give', 'dont', 'dumb', 'wanna', 'retarded', 'fag', 'im', 'bout', 'as', 'ugly', 'dick', 'pussy', 'nigger', 'fuckin', 'shit', 'niggah', 'gotta']
# Load and clean dataset for sarcasm
df_sarcasm = pd.read_csv("Sarcasm_Headlines_Dataset.csv")
df_sarcasm['headline'] = df_sarcasm['headline'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', str(x).lower()))
df_sarcasm['headline'] = df_sarcasm['headline'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stopword]))

# Load and clean dataset for hate span
df_hate_span = pd.read_csv("twitter_data.csv")
df_hate_span['tweet'] = df_hate_span['tweet'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', str(x).lower()))
df_hate_span['tweet'] = df_hate_span['tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stopword]))

# Map labels: 0 and 1 -> Positive class; 2 -> Negative class
df_hate_span['labels'] = df_hate_span['class'].map({0: 1, 1: 1, 2: 0})  # 0 for positive (0 and 1), 1 for negative (2)
y_hate_span = df_hate_span['labels']

# Vectorize data using the same vectorizer for both datasets
vectorizer = CountVectorizer()
X_sarcasm = vectorizer.fit_transform(df_sarcasm['headline'])
X_hate_span = vectorizer.transform(df_hate_span['tweet'])  # Use transform to ensure the same feature set
y_sarcasm = df_sarcasm['is_sarcastic']

# Split data before applying SMOTE
X_train_sarcasm, X_test_sarcasm, y_train_sarcasm, y_test_sarcasm = train_test_split(X_sarcasm, y_sarcasm, test_size=0.2, random_state=42)
X_train_hate_span, X_test_hate_span, y_train_hate_span, y_test_hate_span = train_test_split(X_hate_span, y_hate_span, test_size=0.2, random_state=42)

# Apply SMOTE to the hate span training set
smote = SMOTE(random_state=42)
X_train_hate_span_resampled, y_train_hate_span_resampled = smote.fit_resample(X_train_hate_span, y_train_hate_span)

# Train models
model_sarcasm = MultinomialNB()
model_hate_span = SVC(kernel='linear', probability=True)

model_sarcasm.fit(X_train_sarcasm, y_train_sarcasm)
model_hate_span.fit(X_train_hate_span_resampled, y_train_hate_span_resampled)  # Use resampled data

# Get accuracy and AUC scores
accuracy_sarcasm = accuracy_score(y_test_sarcasm, model_sarcasm.predict(X_test_sarcasm))
accuracy_hate_span = accuracy_score(y_test_hate_span, model_hate_span.predict(X_test_hate_span))

# Calculate AUC scores
auc_sarcasm = roc_auc_score(y_test_sarcasm, model_sarcasm.predict_proba(X_test_sarcasm)[:, 1])
auc_hate_span = roc_auc_score(y_test_hate_span, model_hate_span.predict_proba(X_test_hate_span)[:, 1])

precision_sarcasm = precision_score(y_test_sarcasm, model_sarcasm.predict(X_test_sarcasm),average='macro')
precision_hate_span = precision_score(y_test_hate_span, model_hate_span.predict(X_test_hate_span),average='macro')

recall_sarcasm = recall_score(y_test_sarcasm, model_sarcasm.predict(X_test_sarcasm),average='macro')
recall_hate_span = recall_score(y_test_hate_span, model_hate_span.predict(X_test_hate_span),average='macro')

f1_score_sarcasm = f1_score(y_test_sarcasm, model_sarcasm.predict(X_test_sarcasm),average='macro')
f1_score_hate_span = f1_score(y_test_hate_span, model_hate_span.predict(X_test_hate_span),average='macro')

classification_sarcasm = classification_report(y_test_sarcasm, model_sarcasm.predict(X_test_sarcasm))
classification_hate_span = classification_report(y_test_hate_span, model_hate_span.predict(X_test_hate_span))


def plot_metrics(y_test, y_pred, y_prob, model_name):
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    ax[0].set_title(f"{model_name} Confusion Matrix")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    ax[1].plot(fpr, tpr, color='blue', label=f"AUC = {auc_score:.2f}")
    ax[1].plot([0, 1], [0, 1], color='red', linestyle='--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'ROC Curve - {model_name}')
    ax[1].legend(loc='lower right')

    st.pyplot(fig)

# Streamlit app title
st.title("Sarcasm and Hate Span Detection Application")
st.write(f"**Using the best model for Sarcasm Detection: Multinomial Naive Bayes (Accuracy: {accuracy_sarcasm:.2f}, Precision :{precision_sarcasm:.2f}, Recall :{recall_sarcasm:.2f}, F1 Score :{f1_score_sarcasm:.2f}, AUC: {auc_sarcasm:.2f})**")
st.write("**Classification Report :** ")
st.text(classification_sarcasm)
y_pred_sarcasm = model_sarcasm.predict(X_test_sarcasm)
y_prob_sarcasm = model_sarcasm.predict_proba(X_test_sarcasm)[:, 1]
plot_metrics(y_test_sarcasm, y_pred_sarcasm, y_prob_sarcasm, "Sarcasm Detection")
st.write(f"**Using the best model for Hate Span Detection: Support Vector Classifier (Accuracy: {accuracy_hate_span:.2f},  Precision :{precision_hate_span:.2f}, Recall :{recall_hate_span:.2f}, F1 Score :{f1_score_hate_span:.2f}, AUC: {auc_hate_span:.2f})**")
st.write("**Classification Report :** ")
st.text(classification_hate_span)
y_pred_hate_span = model_hate_span.predict(X_test_hate_span)
y_prob_hate_span = model_hate_span.decision_function(X_test_hate_span)  # SVM requires `decision_function` for ROC
plot_metrics(y_test_hate_span, y_pred_hate_span, y_prob_hate_span, "Hate Span Detection")

def normalize_repeated_characters(text):
    return re.sub(r'(.){2,}', r'', text)  # Replace three or more repeated characters with a single instance

# Prediction function
def analyze_text(input_text):
    input_text = normalize_repeated_characters(input_text)
    input_text_cleaned = re.sub('[%s]' % re.escape(string.punctuation), '', input_text.lower())  # Remove punctuation
    input_text_cleaned = ' '.join([lemmatizer.lemmatize(word) for word in input_text_cleaned.split() if word not in stopword])
    input_vectorized = vectorizer.transform([input_text_cleaned])  # Use the same vectorizer for input

    prediction_sarcasm = model_sarcasm.predict(input_vectorized)[0]
    result_sarcasm = "Sarcastic" if prediction_sarcasm == 1 else "Not Sarcastic"

    prediction_hate_span = model_hate_span.predict(input_vectorized)[0]
    result_hate_span = "Yes" if prediction_hate_span == 1 else "No"

    # Extract sarcasm and hate speech words
    sarcasm_words = [word for word in input_text_cleaned.split() if word in unique_sarcastic_words]
    hate_speech_words = [word for word in input_text_cleaned.split() if word in unique_hate_speech_words]

    st.write(f"**Is it sarcastic? :** {result_sarcasm}")
    st.write(f"**Contains hate speech? :** {result_hate_span}")

    if result_sarcasm == "Sarcastic":
        st.write(f"**Words indicating sarcasm:** {', '.join(sarcasm_words)}")

    if result_hate_span == "Yes":
        st.write(f"**Words indicating hate speech:** {', '.join(hate_speech_words)}")

# Get user input and run analysis
input_text = st.text_input("Enter a sentence to check if it is sarcastic and contains hate speech:")
if input_text:
    analyze_text(input_text)
    
