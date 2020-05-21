selection = input('\n\n1: load_rotten_tomatoes_sentiment_predictino\
_model\n2: Twitter_weather_prediction_model\n3: IMDB_sentiment_predict_model\
\n\nSelection [1,2,3]->')

import load_data
import explore_data
import train_model
import vectorize_data

if int(selection) == 1:
    data = load_data.load_rotten_tomatoes_sentiment_dataset('./data')
    train_model.train_sequence_model(data, './data',save_as='RT')
    print('\n'*5,'-'*75)
    print('Completed!')

elif int(selection) == 2:
    data = load_data.load_tweet_weather_topic_classification_dataset('./data')
    train_model.train_sequence_model(data, './data',save_as='TW')
    print('\n'*5,'-'*75)
    print('Completed!')

elif int(selection) == 3:
    data = load_data.load_imdb_sentiment_analysis_dataset('./data')
    train_model.train_sequence_model(data, './data',save_as='IMDB')
    print('\n'*5,'-'*75)
    print('Completed!')

else:
    print('Invalid selection ...')
