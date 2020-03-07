# Disaster Response Pipeline Project
### Installation
The code should run with no issues using Python versions 3.X. The main libraries used include padas,sklearn and nltk

### Project Motivation
The objective of the project is to illustrate how a machine learning pipeline could be used for natural language processing. It fundamentally includes following steps.
1. Read labelled text from csv files and process them to create a denormalised data structure.
2. Train a classification model using the GridSearchCV and export the model
3. Build a web application to utilize the model and visualize the data.
### File Descriptions
1. 'data' directory contains the python script 'process_data.py' related to the ETL process. The csv files 'disaster_categories.csv' and 'disaster_messages.csv' would be the data inputs to the ETL process. The process data is written to the SQLLite db DisasterResponse.db
2. 'models' directory contains the 'train_classifer.py' which would train the model. The resultant model would be written to the file 'classifier.pkl' 
3. 'app' directory contains the flask web application files. The web application provides a dashboard including charts indicating the distribution of the data set. It also provides a utility to scan messages and output the relavent category based on the trained model.
### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl dev`

2. Note that for the model building there is an additional profile parameter passed. The accepted parameter values are 'dev' and 'prod'. This is introduced to build a model in environments where there are resource constraints. When running under the 'dev' profile, only 5000 first records would be used for training and also the GridSeachCV would consider lesser number of paramaters. 

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/
### Demo site
A public demo site is available at http://35.192.151.198:3001/ This is deployed in Google Cloud. 
### Results
Following are the results of the classification report. Note that only first 1000 records were considered due to performance reasons. 
| category               | precision | recall | f1-score | support | remarks       |
|------------------------|-----------|--------|----------|---------|---------------|
| related                | 0         | 0      | 0        | 33      | poor          |
| request                | 0.44      | 0.22   | 0.29     | 88      | poor          |
| offer                  | 0.99      | 1      | 1        | 199     |               |
| aid_related            | 0.46      | 0.12   | 0.2      | 88      | poor          |
| medical_help           | 0.9       | 1      | 0.95     | 180     |               |
| medical_products       | 0.95      | 0.99   | 0.97     | 190     |               |
| search_and_rescue      | 0.96      | 1      | 0.98     | 192     |               |
| security               | 0.96      | 1      | 0.98     | 193     |               |
| military               | 1         | 1      | 1        | 200     |               |
| child_alone            | 1         | 1      | 1        | 200     |               |
| water                  | 0.82      | 0.99   | 0.9      | 165     | low precision |
| food                   | 0.75      | 0.95   | 0.84     | 152     | low precision |
| shelter                | 0.86      | 1      | 0.92     | 172     | low precision |
| clothing               | 0.97      | 1      | 0.98     | 194     |               |
| money                  | 0.98      | 1      | 0.99     | 196     |               |
| missing_people         | 0.99      | 1      | 0.99     | 198     |               |
| refugees               | 0.97      | 1      | 0.99     | 195     |               |
| death                  | 0.98      | 1      | 0.99     | 196     |               |
| other_aid              | 0.79      | 0.99   | 0.88     | 159     |               |
| infrastructure_related | 0.95      | 1      | 0.98     | 191     |               |
| transport              | 0.97      | 1      | 0.98     | 194     |               |
| buildings              | 0.94      | 1      | 0.97     | 188     |               |
| electricity            | 0.99      | 1      | 0.99     | 198     |               |
| tools                  | 0.99      | 1      | 0.99     | 198     |               |
| hospitals              | 0.99      | 1      | 0.99     | 198     |               |
| shops                  | 0.99      | 1      | 1        | 199     |               |
| aid_centers            | 0.99      | 1      | 0.99     | 198     |               |
| other_infrastructure   | 0.97      | 1      | 0.99     | 195     |               |
| weather_related        | 0.89      | 1      | 0.94     | 177     |               |
| floods                 | 0.97      | 1      | 0.98     | 194     |               |
| storm                  | 0.97      | 1      | 0.98     | 194     |               |
| fire                   | 0.99      | 1      | 1        | 199     |               |
| earthquake             | 0.96      | 1      | 0.98     | 193     |               |
| cold                   | 0.98      | 1      | 0.99     | 197     |               |
| other_weather          | 0.98      | 1      | 0.99     | 197     |               |
| direct_report          | 0.52      | 0.37   | 0.43     | 94      | poor          |
### Future work
#### Optimisation tips
When training the models in multi-core infrastructure, significant gains can be achieved by setting the n_jobs parameter for the classifier. Setting this value to -1 would ensure that all available cores would be used for cross validation.
```
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))])
```
#### Dataset is imbalance
There seems to be an imbalance interms of the attributes as some of the attributes have limited number of examples. The attributes flagged as 'poor' has low precision and low recall rates where as attributes flagged as 'low precision' has recorded low precision rates compared to more better performing attributes. These attributes have lower level of examples which seems to have contributed to the low precision and recall values. So the predictions related to these categories will have low accuracy. 
### Licensing, Authors, Acknowledgements
Creadit must go to Figure Eight (www.figure-eight.com) for providing the labelled data set and Udacity for providing the code outline. You are most welcome to use it!

