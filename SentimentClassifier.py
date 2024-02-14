import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import gensim.downloader

# Load train and test data
train_data = pd.read_csv(r"C:\Users\KAVYA K\Downloads\train.csv")
test_data = pd.read_csv(r"C:\Users\KAVYA K\Downloads\test.csv", header=None)

# Drop NA values
train_data.dropna(inplace=True)

# Preprocessing function for tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.lower()
    return tweet

# Apply preprocessing to train and test tweets
train_data['clean_text'] = train_data['Tweets'].apply(preprocess_tweet)
test_data[0] = test_data[0].apply(preprocess_tweet)

# Tokenization
max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_data['clean_text'])
X_train_seq = tokenizer.texts_to_sequences(train_data['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(test_data[0])

# Padding sequences
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_data['label'])

# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_pad, y_train_encoded, test_size=0.2, random_state=42)

# Download and load pre-trained Word2Vec embeddings
word_vectors = gensim.downloader.load('word2vec-google-news-300')

# Create an embedding matrix
embedding_matrix = np.zeros((max_features, word_vectors.vector_size))
for word, i in tokenizer.word_index.items():
    if i < max_features and word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

# Build LSTM Model with pre-trained embeddings
embedding_dim = word_vectors.vector_size
lstm_out = 196

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, 
                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    input_length=max_len,
                    trainable=False))  # Set to False to keep embeddings fixed during training
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Training the model
batch_size = 32
epochs = 10

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

# Evaluate model on test data
y_pred_probabilities = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probabilities, axis=-1)

# If you have test labels for evaluation
# Encode test labels and calculate accuracy
y_test_encoded = label_encoder.transform(test_data[1])
test_accuracy = accuracy_score(y_test_encoded, y_pred)
print("Test Accuracy:", test_accuracy)

# If you don't have test labels for evaluation
# Write y_pred to a file for submission or further analysis
# For example, assuming y_pred is a list of predicted labels
# with open('predicted_labels.csv', 'w') as f:
#     for label in y_pred:
#         f.write(f"{label}\n")
