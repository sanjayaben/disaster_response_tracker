import sys
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def load_data(database_filepath,profile):
    '''
    INPUT
    database_filepath - SQLLite database file to read data from
    
    OUTPUT
    Features, Labels and category names
    
    Loads the data and splits into labels and features
    '''
    #Load data from the database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from messages_flattened", engine)
    if profile == 'dev':
        print('Running dev mode !!!!!!!!!!')
        df = df.head(5000)
    #drop unnessasay columns
    df = df.drop(columns=["id","original", "genre"])
    '''Some rows contain all null values for the label data. 
       There are no partial missing values.
       Hence null rows can be dropped as follows'''
    df = df.dropna(axis='rows')

    categories = df.drop(columns=["message"])
    X = df.message.values
    y = categories.values
    category_names = categories.columns
    return X,y,category_names


def tokenize(text):
    '''
    INPUT
    text - text to be tokenized
    
    OUTPUT
    Generated tokens
    
    Tokenizes a given text by splitting into words, converting to lower case, removing white spaces and lemmatizing
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    #remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    # tokenize text
    tokens = word_tokenize(text)
   
    clean_tokens = []
    for tok in tokens:
        #Lemmatize the token, convert to lower case and remove white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        #Remove stop words
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(profile):
    '''
    INPUT
    None
    
    OUTPUT
    GridSearch model
    
    This function does the following
    1. Creates a pipeline that includes a CountVectorizer, TfidfTransformer and a RandomForestClassifier
    2. Define a set of parameter value ranges across all estimaters
    3. Create and return a GridSearchCV
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {}
    
    if profile == 'dev':
        parameters = {
            'vect__max_df': (0.5, 1.0),
            'tfidf__use_idf': (True, False),
            'clf__estimator__min_samples_split': [2, 4]}
       
    else:
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 1.0),
            'vect__max_features': (None, 5000),
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [50, 100],
            'clf__estimator__min_samples_split': [2, 4]}
        

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - the model
    X_test - test data set for features
    Y_test - test data set for labels
    category_names - category names
    
    OUTPUT
    None
    
    This function will printout the classification report to each category
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    #for each column print the clasification report
    for index,col in enumerate(category_names): 
        print(classification_report(Y_test[:,index], y_pred[:,index],target_names=[col],labels=[0, 1]))



def save_model(model, model_filepath):
    '''
    INPUT
    model - trained model
    model_filepath - path to store the model
    '''
    model_filepath = model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, profile = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath,profile)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(profile)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()