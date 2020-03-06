import sys
import logging
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - file path for the messages data
    categories_filepath - file path for the categories data
    
    OUTPUT
    df - dataframe containing the loaded and process data
    
    This function loads the data from the specified files and perform the following operations
    1. Merge the two data sets on the "id" column
    2. Process the categories data to extract the 0, 1 values for each category and introduce meaningful column headers
    
    '''
    #Load Data
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    #Merge two data sets on id
    df = pd.merge(messages, categories, on='id')
    #process categories and return dataframe
    return process_categories(df,categories)

def process_categories(df,categories):
    '''
    INPUT
    df - dataframe from previous steps
    categories - categories data frame
    
    OUTPUT
    df - dataframe with the processed category columns
    This function would create column for each category entry and add the respective values 0 or 1. 
    Also would add meaningful column headers derived from the data
    '''
    #Split the categories column into individual columns
    categories = categories['categories'].str.split(";",expand=True)
    #Take the first data row
    row = categories.iloc[0]
    #Take out the last two digits that correspod to the value and consider rest for the column names
    category_colnames = row.apply(lambda col_name : col_name[:-2])
    #Replace column names
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda row_val : row_val[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda val : int(val))
    #drop the existing categiries column and merge the new columns
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    '''
    INPUT
    df - dataframe
    
    OUTPUT
    df - dataframe with dulicates removed
    
    This function removes the duplicates. Duplicates are identified by considering all columns
    '''
    logging.info("Total records : ", len(df))
    logging.info("Duplicate count : ", len(df[df.duplicated(keep='last')]))
    df = df.drop_duplicates(keep='last')
    logging.info("New Duplicate count : ", len(df[df.duplicated(keep=False)]))
    return df

def save_data(df, database_filename):
    '''
    INPUT
    df - processed dataframe
    database_filename - filename for the database to be created
    
    OUTPUT
    none
    
    This function stores the given data from in an SQLLite database file
    '''
    engine = create_engine('sqlite:///' + database_filename )
    df.to_sql('messages_flattened', con=engine, if_exists='replace')  
    
def test_data(database_file,df):
    conn = sqlite3.connect(database_file)
    read_data_size = len(pd.read_sql("select * from messages_flattened", conn))
    df_size = len(df)
    assert(read_data_size==df_size)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
        print('Testing....')
        test_data(database_filepath,df)
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()