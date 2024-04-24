from gravityai import gravityai as grav

# pickling python converted into a bin ary file
# serializing: it involves converting complex data structures, such as objects or arrays, into a byte stream or a string that can be written to a file, sent over a network, or stored in a database.

import pickle

# library for data manupulation
import pandas as pd

# create pickle file
model = pickle.load(open ('financial_text_classifier.pkl', 'rb'))
# tfidf OR term frequency inverse document frequency - machine learning term
# help with word freq. in our corpus    https://en.wikipedia.org/wiki/Tf%E2%80%93idf
tfidf_vectorizer = pickle.load(open ('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open ('financial_text_encoder.pkl', 'rb'))

# function
def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizer.transform(input_df['body'])
    # predict the classes
    predicitions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predicitions)
    # save into a csv
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)

# pass the function into gravity ai helper
grav.wait_for_request(process)

# issue with .py in gravityai
# Check the version of scikit-learn that was used to create the pickled model by running !pip show scikit-learn in the environment where the model was trained.