import sys
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import sqlalchemy

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """ Returns X(array), Y(array) and category_names(list).
    Parameters:
        database_filepath (str): The string which is the database path to load the dataframe.

    Returns:
        X(array): The array which are x variables.
        Y(array): The array which are y variables.
        category_names(list): The list which contains category names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_query("select * from Messages;", conn).fillna(0)
    X = df['message'].str.strip().map(lambda x: re.sub('[!@#$-]', '',x))
    Y = df.iloc[:, 4:].values
    category_names = list(df.iloc[:, 4:].columns)
    return X, Y, category_names


def tokenize(text):
    """ Returns the tokenized text list.
    Parameters:
        text (str): The string which is the text to be tokenized.

    Returns:
        clean_tokens(list): The list which contains the tokenized text.
    """
    # delete url in text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ Returns the GridSearchCV model.
    No Parameter

    Returns:
        cv(GridSearchCV object): The object which contains the pipeline.
    """
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # build grid search model
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate model.
    Parameters:
        model (object): The model to evaluate.
        X_test (array): The array which are x test variables.
        Y_test (array): The array which are y test variables.
        category_names(list): The list which contains category names.

    No Return
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i] + ": ")
        print(classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True)['weighted avg'])


def save_model(model, model_filepath):
    """ Save model.
    Parameters:
        model (object): The model to save.
        model_filepath (str): The string which is the path to save the model.

    No Return
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # split data to train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # build model
        print('Building model...')
        model = build_model()

        # train model
        print('Training model...')
        model.fit(X_train, Y_train)

        # evaluate model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save model
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