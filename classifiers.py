import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def app():
    df = pd.read_csv("nb.csv")
    df = df.dropna()
    st.write(df.head(10))
    st.write("The above dataset will be split in 70-30 train test data.")
    

    #X = essay, Y = gender
    X_train, X_test, y_train, y_test = train_test_split(df['My Self Summary'], df['Gender'], test_size=0.30, random_state=1)
    

    tf_idf_vec = TfidfVectorizer()
    tf_idf = tf_idf_vec.fit_transform(X_train)
    st.write("Some Features(aka words) and their Tf-Idf score:")
    st.write(tf_idf_vec.get_feature_names_out()[:20])
    st.write(tf_idf_vec.get_feature_names_out()[:-20])
    st.write(tf_idf_vec.get_feature_names_out()[7000:7020])
