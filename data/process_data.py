import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Returns the dataframe from raw data.
    Parameters:
        messages_filepath (str): The string which is the raw data(messages) path.
        categories_filepath (str): The string which is the raw data(categories) path.

    Returns:
        df(dataframe): The dataframe which loaded from raw data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner')
    return df


def clean_data(df):
    """ Returns the cleaned dataframe.
    Parameters:
        df (dataframe): The dataframe which loaded from raw data.

    Returns:
        df(dataframe): The cleaned dataframe to save.
    """
    # split categories
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0].values
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column], downcast='signed')
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1).drop_duplicates()
    return df


def save_data(df, database_filename):
    """ Save the cleaned dataframe.
    Parameters:
        df (dataframe): The cleaned dataframe to save.
        database_filename (str): The string which is the path to save the dataframe.

    No Return

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean data
        print('Cleaning data...')
        df = clean_data(df)

        # save data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()