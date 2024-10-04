import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from keras.utils import to_categorical

# Load and preprocess data
def load_data():
    training_data = pd.read_csv(r'E:\Soundari akka\Data\Training.csv')
    testing_data = pd.read_csv(r'E:\Soundari akka\Data\Testing.csv')

    symptoms_train = training_data["prognosis"]
    symptoms_test = testing_data["prognosis"]

    text_tokenizer = Tokenizer(num_words=10000)
    text_tokenizer.fit_on_texts(symptoms_train)

    symptoms_seq_train = text_tokenizer.texts_to_sequences(symptoms_train)
    symptoms_seq_test = text_tokenizer.texts_to_sequences(symptoms_test)

    max_len = max(len(seq) for seq in symptoms_seq_train + symptoms_seq_test)
    symptoms_padded_train = pad_sequences(symptoms_seq_train, maxlen=max_len)
    symptoms_padded_test = pad_sequences(symptoms_seq_test, maxlen=max_len)

    labels_train = training_data["prognosis"]
    labels_test = testing_data["prognosis"]

    le = preprocessing.LabelEncoder()
    le.fit(labels_train)
    labels_encoded_train = le.transform(labels_train)
    labels_encoded_test = le.transform(labels_test)

    x_train, x_test, y_train, y_test = train_test_split(
        symptoms_padded_train, labels_encoded_train, test_size=0.33, random_state=42
    )

    return (
        x_train, x_test, y_train, y_test,
        symptoms_padded_test, labels_encoded_test,
        le, max_len, text_tokenizer
    )

# Build LSTM model
def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=64):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(len(le.classes_), activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

# Train LSTM model
def train_lstm_model(model, x_train, y_train, epochs=25):
    # One-hot encode the target data
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Train LSTM model with one-hot encoded target data
    model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=32, validation_split=0.1)

# Build decision tree classifier
def build_decision_tree(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf

# Predict disease
def predict_disease(symptoms, model_type, tokenizer):
    symptoms_seq = tokenizer.texts_to_sequences([symptoms])
    symptoms_padded = pad_sequences(symptoms_seq, maxlen=max_len)

    if model_type == "lstm":
        prediction_probs = lstm_model.predict(symptoms_padded)[0]  # Get prediction probabilities
    elif model_type == "decision_tree":
        prediction_probs = decision_tree_model.predict_proba(symptoms_padded)[0]  # Get prediction probabilities
    else:
        return "Invalid model type"

    predicted_label_index = prediction_probs.argmax()  # Get index of maximum probability
    predicted_label = le.inverse_transform([predicted_label_index])[0]  # Inverse transform to get label
    return predicted_label


# Load data
x_train, x_test, y_train, y_test, symptoms_padded_test, labels_encoded_test, le, max_len, text_tokenizer = load_data()

# Build and train LSTM model
lstm_model = build_lstm_model(vocab_size=10000)
train_lstm_model(lstm_model, x_train, y_train)

# Build decision tree model
decision_tree_model = build_decision_tree(x_train, y_train)

# Flask app
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    symptoms = request.form['symptoms']
    model_type = request.form['model_type']
    predicted_label = predict_disease(symptoms, model_type, text_tokenizer)
    return render_template('result.html', symptoms=symptoms, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
