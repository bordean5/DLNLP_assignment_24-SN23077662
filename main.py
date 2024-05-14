from src import *
from tensorflow.keras.utils import to_categorical

def main(train_bert=False):
    """
    Main function to run the whole project.
    It loads the data, preprocesses it, trains, and evaluates Naive Bayes and LSTM models.
    The tuned BERT model is by default loaded from weights. If you want to train it, set train_bert to True.

    Args:
        train_bert (bool): If True, trains the BERT model and saves the weights.
                           If False, loads the BERT model weights from the specified path.
    """
    
    # Load and preprocess data
    train_data = load_data('./Datasets/train.tsv.zip')
    clean_text = preprocessing(train_data)
    target = to_categorical(train_data['Sentiment'].values)#convert labels to one-hot encodings

    #split into train, valid, test datasets
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(clean_text, target)
    
    # Train and evaluate Naive Bayes model
    tfid_vectorizer, nb_pipeline = create_nbmodel()
    print('Evaluation of Naive Bayes:')
    print("\n")
    nb_model = train_nbmodel(tfid_vectorizer, nb_pipeline, X_train, np.argmax(y_train, axis=1))
    evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')

    # Train and evaluate LSTM model
    LSTM_model = create_LSTMmodel(X_train)
    print("LSTM training:")
    LSTM_history, LSTM_model = train_LSTMmodel(LSTM_model, X_train, y_train, X_val, y_val)
    print('Evaluation of LSTM')
    print("\n")
    evaluate_model(LSTM_model, X_test, y_test, 'LSTM')
    plot_history(LSTM_history)
    
    # Create and train or load BERT model
    tune_bert = create_bertmodel()
    #encode the data with BertTokenizer
    bert_train, bert_val, bert_test = bert_encoding(X_train, X_val, X_test, y_train, y_val, y_test) 


    if train_bert:
        # Train BERT model
        bert_history, tune_bert = train_bertmodel(tune_bert, bert_train, bert_val)
        # Save weights
        tune_bert.save_weights('Model/newBert_weights.h5')
    else:
        # Load pre-trained weights
        tune_bert.load_weights('Model/Bert_weights.h5')
    
    # Evaluate BERT model
    print('Evaluation of Bert')
    print("\n")
    evaluate_model(tune_bert, bert_test, y_test, 'tuned Bert')


if __name__ == "__main__":
    main(train_bert=False)  # Set to True if you want to train the BERT model