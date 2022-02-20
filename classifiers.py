import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def app():
    df = pd.read_csv("nb.csv")
    df = df.dropna()
    st.write(df.head(10))
    st.write("The above dataset will be split in 70-30 train test data.")
    

    #X = essay, Y = gender
    X_train, X_test, y_train, y_test = train_test_split(df['My Self Summary'], df['Gender'], test_size=0.30, random_state=1)
    

    tf_idf_vec = TfidfVectorizer()
    tf_idf = tf_idf_vec.fit_transform(X_train)
    st.subheader("Some Features(aka Words) in Tf-Idf Matrix")
    st.write(tf_idf_vec.get_feature_names())[1000:1100]
    nb_clf = MultinomialNB()
    nb_clf.fit(tf_idf, y_train)

    st.subheader("Naive Bayes Predictions vs Original Gender")
    predictions = nb_clf.predict(tf_idf_vec.transform(X_test))
    st.write(pd.DataFrame({"Predictions":predictions, "Original":y_test}).head(20))
    
    #st.subheader("Performance")
    #bar chart for accuracy precision recall
