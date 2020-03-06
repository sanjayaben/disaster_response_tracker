import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_flattened', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    #disaster types analysis
    disaster_counts = df[['floods','storm','fire','earthquake','cold','other_weather']].sum(axis = 0, skipna = True)
    disaster_names = list(disaster_counts.index)
    
    #type of request analysis
    type_counts = df[[ 'aid_related', 'medical_help','water','food','shelter','clothing','money']].sum(axis = 0, skipna = True)
    type_names = list(type_counts.index)
    
    #all categories analysis
    all_counts = df.drop(columns=['index','id','message','genre']).sum(axis = 0, skipna = True)
    all_names = list(all_counts.index) 
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [        
        create_report_pie(disaster_names,disaster_counts,'Distribution by Type of Disaster','Type of Disaster','Count'),
        create_report_pie(type_names,type_counts,'Distribution by Type of Request','Type of Request','Count'),
        create_report_bar(genre_names,genre_counts,'Distribution of Message Genres','Genre','Count'),   
        create_report_bar(all_names,all_counts,'Distribution of all Message Categories','Category','Count') 
    ]
     
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

def create_report_bar(labels,values,chart_title,x_title,y_title):
    return {
            'data': [
                Bar(
                    x=labels,
                    y=values 
                )
            ],

            'layout': {
                'title': chart_title,
                'yaxis': {
                    'title': y_title
                },
                'xaxis': {
                    'title': x_title
                }
            }
        }
def create_report_pie(labels,values,chart_title,x_title,y_title):
    return {
            'data': [
                Pie(
                    labels=labels,
                    values=values
                )
            ],

            'layout': {
                'title': chart_title,
                'yaxis': {
                    'title': y_title
                },
                'xaxis': {
                    'title': x_title
                }
            }
        }
    
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()