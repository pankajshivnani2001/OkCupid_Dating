import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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
    st.write(tf_idf_vec.get_feature_names()[5000:5100])
    nb_clf = MultinomialNB()
    nb_clf.fit(tf_idf, y_train)

    st.subheader("A few Naive Bayes Predictions vs Original Gender")
    predictions = nb_clf.predict(tf_idf_vec.transform(X_test))
    st.write(pd.DataFrame({"Essay": X_test, "Predictions":predictions, "Original":y_test}).head(20))
    
    st.subheader("Performance")

   
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')

    st.subheader("Naive Bayes Performance")
    st.write("accuracy:\t", accuracy)
    st.write("precision:\t", precision)
    st.write("recall:\t\t", recall)

    rf_clf = RandomForestClassifier(n_estimators = 100)
    rf_clf.fit(tf_idf, y_train)
    predictions = rf_clf.predict(tf_idf_vec.transform(X_test))
    t.write(pd.DataFrame({"Essay": X_test, "Predictions":predictions, "Original":y_test}).head(20))
    
    st.subheader("Performance")

   
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')

    st.subheader("Random Forest Performance")
    st.write("accuracy:\t", accuracy)
    st.write("precision:\t", precision)
    st.write("recall:\t\t", recall)

