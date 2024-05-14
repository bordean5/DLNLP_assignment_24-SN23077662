import os
import spacy 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import TFAutoModel



def load_data(path):
    """
      load the data with file path 
    Args:
        path: work dir
    Returns:
        data: csv dataset
    """
    data = pd.read_csv(path, sep='\t', compression='zip') 

    return data


def preprocessing(document):
    """
    Preprocesses the text data by performing cleaning operations.
    Tokenize, Lemmatize, Remove non alphabet character
    
    Args:
        data (DataFrame): The input data with a 'Phrase' column.
        
    Returns:
        list: Cleaned text data.
    """
    nlp = spacy.load('en_core_web_sm') #trypython -m spacy download en if error occurs
    preprocessed_texts = []
    
    for text in document['Phrase'].values:
        text = str(text)
        result = [token.lemma_ for token in nlp(text) if token.is_alpha]
        preprocessed_texts.append(" ".join(result))
        
    return preprocessed_texts

def data_split(text, target):
    """
    Splits the data into training, validation, and test sets.
    
    Args:
        text (list): Cleaned text data.
        target (array): Target labels.
        
    Returns:
        tuple: Split data (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        text,
        target,
        train_size=0.9,
        stratify=target,
        shuffle = True,
        random_state = 42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid,
        y_train_valid,
        train_size=0.85,
        stratify=y_train_valid,
        shuffle = True,
        random_state = 42
    ) 

    X_train = np.array(X_train, dtype=object)
    y_train = np.array(y_train)
    X_val = np.array(X_val, dtype=object)
    y_val = np.array(y_val)
    X_test = np.array(X_test, dtype=object)
    y_test = np.array(y_test)

    return X_train, X_val,X_test, y_train,y_val,y_test

def create_nbmodel():
    """
    Creates a Naive Bayes model pipeline with TF-IDF vectorization.
    
    Returns:
        TF-IDF vectorizer and Naive Bayes model pipeline.
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Create a Naive Bayes classifier pipeline
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    
    return vectorizer, nb_pipeline

def train_nbmodel(vectorizer, nb_pipeline, X_train, y_train):
    """
    Trains the Naive Bayes model.
    
    Args:
        tfid_vectorizer (TfidfVectorizer): TF-IDF vectorizer.
        nb_pipeline (Pipeline): Naive Bayes model pipeline.
        X_train : Training text data.
        y_train : Training labels.
        
    Returns:
        Pipeline: Trained Naive Bayes model pipeline.
    """
    # Fit the TF-IDF vectorizer and transform the training data
    tfidf_text = vectorizer.fit_transform(X_train)
    
    # Train the Naive Bayes classifier pipeline
    nb_pipeline.fit(X_train, y_train)
    
    train_accuracy = nb_pipeline.score(X_train, y_train)
    print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy))
    
    return nb_pipeline

def create_LSTMmodel(X_train):
    """
    Creates an LSTM model for text classification.
    
    Args:
        X_train : Training text data.
        
    Returns:
        Model: Compiled LSTM model.
    """
    unique_words = set()
    len_max = 0

    #get unique words and maximum length of tokens
    for sent in X_train:
        tokens = sent.split()
        unique_words.update(tokens)
    
        if len_max < len(tokens):
            len_max = len(tokens)

    vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=len(unique_words),
    output_mode='int',
    output_sequence_length=len_max
    )
    vectorize_layer.adapt(list(unique_words))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(tf.keras.layers.Embedding(len(unique_words), 300, input_length=len_max))
    model.add(tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        metrics=['accuracy']
    )
    return model

def train_LSTMmodel(model, X_train, y_train, X_val, y_val):
    """
    Trains the LSTM model.
    
    Args:
        model (Model): Compiled LSTM model.
        X_train (array): Training text data.
        y_train (array): Training labels.
        X_val (array): Validation text data.
        y_val (array): Validation labels.
        
    Returns:
        tuple: Training history and trained LSTM model.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta = 0.001, mode = 'min', monitor='val_loss', patience = 2)
    callback = [early_stopping]

    history = model.fit(
        X_train,
        y_train, 
        validation_data=(
            X_val,
            y_val
        ), 
        epochs=6, 
        batch_size=256, 
        verbose=1,
        callbacks=callback
    )

    return history, model

def create_bertmodel():
    """
    Creates a BERT model for text classification.
    
    Returns:
        Model: Compiled BERT model.
    """
    len_max=60
    #define inputs: ids and attention mask 
    input_ids = Input(shape=(len_max,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(len_max,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    bert = TFAutoModel.from_pretrained('bert-base-uncased')
    embeddings = bert.bert(inputs)[1]

    # convert bert embeddings into 5 output classes
    output = Flatten()(embeddings)
    output = Dense(256, activation='relu')(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(5, activation='softmax', name='outputs')(output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-05),
    metrics=['accuracy']
    )

    return model

def bert_encoding(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Encodes text data for BERT model.

    Args:
        X_train (array): Training text data.
        X_val (array): Validation text data.
        X_test (array): Test text data.
        y_train (array): Training labels.
        y_val (array): Validation labels.
        y_test (array): Test labels.

    Returns:
        tuple: Encoded training, validation, and test datasets.
    """
    
    def encode_texts(text_list, tokenizer, max_length=60):
        return tokenizer(text_list, 
                     add_special_tokens=True,  
                     max_length=max_length,    
                     truncation=True,         
                     padding='max_length',    
                     return_tensors='tf') 

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_train_list=X_train.tolist()
    X_val_list=X_val.tolist()
    X_test_list=X_test.tolist()

    train_encodings = encode_texts(X_train_list, tokenizer)
    val_encodings = encode_texts(X_val_list, tokenizer)
    test_encodings = encode_texts(X_test_list, tokenizer)

    train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']}, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask']}, y_val)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}).batch(32)

    return train_dataset, val_dataset, test_dataset

def train_bertmodel(model, train_dataset, val_dataset):
    """
    Trains the BERT model.

    Args:
        model (Model): Compiled BERT model.
        train_dataset (tf.data.Dataset): Encoded training dataset.
        val_dataset (tf.data.Dataset): Encoded validation dataset.

    Returns:
        tuple: Training history and trained BERT model.
    """
    
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    return history, model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluates the model on test data.
    accuracy, precision, recall, F1 score, classification report
    plot and save the confusion matrix
    
    Args:
        model (Model or Pipeline): The trained model.
        X_test: Test iput data
        y_test (array): Test labels.
        model_name (str): Name of the model being evaluated.
        
    Prints:
        Evaluation results including accuracy, precision, recall, F1 score, classification report.
    """
    # Predict the labels for the test data
    y_pred = model.predict(X_test)
    
    # If the predictions are probabilities, convert them to label indices
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        label_pred = np.argmax(y_pred, axis=1)
    else:
        label_pred = y_pred

    # Convert the true labels to indices if they are in one-hot encoding format
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_indices, label_pred)
    f1 = f1_score(y_test_indices, label_pred, average='weighted')
    recall = recall_score(y_test_indices, label_pred, average='weighted')
    precision = precision_score(y_test_indices, label_pred, average='weighted')

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1)
    print(classification_report(y_test_indices, label_pred))

    confu_matrix = confusion_matrix(y_test_indices, label_pred)
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=range(0,5), yticklabels=range(0,5))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    output_dir = 'Images'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir , f'{model_name}_confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()

def plot_history(history):
    """
    Plots training and validation accuracy and loss.
    
    Args:
        history (History): Training history.
    """

    # Extracting accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    # Create subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_style("white")
    plt.suptitle('Train history', size = 15)
    # Plotting training and validation accuracy
    ax1.plot(epochs, acc, "r--", label = "Training acc")
    ax1.plot(epochs, val_acc, "b-", label = "Validation acc")
    ax1.set_title("Training and validation acc")
    ax1.legend()
    ax1.grid(True, linestyle='--', linewidth=0.5)
    # Plotting training and validation loss
    ax2.plot(epochs, loss, "r--", label = "Training loss")
    ax2.plot(epochs, val_loss, "b-", label = "Validation loss")
    ax2.set_title("Training and validation loss")
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)

    output_dir = 'Images'
    os.makedirs(output_dir, exist_ok=True)
    # Save the plot
    output_path = os.path.join(output_dir , f'training history.png')
    plt.savefig(output_path)
    plt.close()


