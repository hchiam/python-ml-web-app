"""
Setup: (use pip or pip3)
  pip3 install streamlit
  pip3 install pandas
  pip3 install -U scikit-learn
Run:
  streamlit run iris-ml-app.py
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


def run():
    data_features = user_input_features()
    prediction = make_predictions(data_features)
    set_up_the_ui(prediction, data_features)


def user_input_features():
    # create sidebar sliders in the UI and get data from them:
    # (last parameter is default value)
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    # consolidate that data:
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    # get features from pandas:
    features = pd.DataFrame(data, index=[0])
    return features


def make_predictions(data_features):
    iris_data_set = datasets.load_iris()
    features = iris_data_set.data
    labels = iris_data_set.target

    classifier = RandomForestClassifier()
    classifier.fit(features, labels)

    names = ', '.join(iris_data_set.target_names) + ':'

    prediction_index = classifier.predict(data_features)
    prediction_label = iris_data_set.target_names[prediction_index]
    prediction_probability = classifier.predict_proba(data_features)

    output = {
        'label': prediction_label,
        'names_to_choose_from': names,
        'probabilities': prediction_probability
    }

    return output


def set_up_the_ui(prediction, data_features):
    st.write("""
    # Simple Iris Flower Prediction App
    This app predicts the **Iris flower** type!
    """)

    st.sidebar.header('User Input Parameters')

    st.subheader('User Input parameters')
    st.write(data_features)

    st.subheader('Prediction')
    st.write(prediction['label'])

    st.subheader('Prediction Probability')
    st.write(prediction['names_to_choose_from'])
    st.write(prediction['probabilities'])


run()
