import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

def app():
    st.header("Predicting Gender Using Self-Summary Essay")
    df = pd.read_csv("nb.csv")
    df = df.dropna()
    df.drop(columns=df.columns[0], axis=1, inplace=True)
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

   
    nb_accuracy = accuracy_score(y_test, predictions)
    nb_precision = precision_score(y_test, predictions, average='weighted')
    nb_f1 = f1_score(y_test, predictions, average='weighted')

    st.subheader("Naive Bayes Performance")
    st.write("accuracy:", nb_accuracy)
    st.write("precision:", nb_precision)
    st.write("F1-Score:", nb_f1)
    conf_mat = confusion_matrix(y_test, predictions, labels=["m", "f"])
    fig = go.Figure(data=go.Heatmap(
        z=conf_mat,
        x=['Male', 'Female'],
        y=['Male', 'Female'],
        hoverongaps = False),
        layout = layout)
    st.plotly_chart(fig)


    st.subheader("A Few Random Forest Predictions vs Original Gender")
    rf_clf = RandomForestClassifier(n_estimators = 20)
    rf_clf.fit(tf_idf, y_train)
    predictions = rf_clf.predict(tf_idf_vec.transform(X_test))
    st.write(pd.DataFrame({"Essay": X_test, "Predictions":predictions, "Original":y_test}).head(20))


    rf_accuracy = accuracy_score(y_test, predictions)
    rf_precision = precision_score(y_test, predictions, average='weighted')
    rf_f1 = f1_score(y_test, predictions, average='weighted')
    

    
    st.subheader("Random Forest Performance")
    st.write("accuracy:", rf_accuracy)
    st.write("precision:", rf_precision)
    st.write("F1-Score:", rf_f1)

    layout = {
        "title": "Confusion Matrix", 
        "xaxis": {"title": "Predicted value"}, 
        "yaxis": {"title": "Real value"}
    }
    conf_mat = confusion_matrix(y_test, predictions, labels=["m", "f"])
    fig = go.Figure(data=go.Heatmap(
            z=conf_mat,
            x=['Male', 'Female'],
            y=['Male', 'Female'],
            hoverongaps = False),
            layout = layout)
    st.plotly_chart(fig)
    st.write("Note that we are only using n_estimators=20 and that may affect the performance of Random Forest. We can increase the performance by increasing the number of trees but the time to train the model will be unreasonable for a web app. Tests conducted showed tha even for 1000 n_estimators, we get a maximum accuracy of 0.66, which is comaparable to Multinomial Naive Bayes")



    st.subheader("Comparing Performance")
    fig = go.Figure(data=[
    go.Bar(name='Naive Bayes', x=["Accuracy", "Precision", "F1-Score"], y=[nb_accuracy, nb_precision, nb_f1]),
    go.Bar(name='Random Forest', x=["Accuracy", "Precision", "F1-Score"], y=[rf_accuracy, rf_precision, rf_f1])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)

    st.subheader("Test Our Model")
    st.write("Enter your Self-Summary and check if the model correctly predicts your Gender")
    text = st.text_input("Your Self-Summary")
    tf_idf_text  = tf_idf_vec.transform([text])
    nb_prediction = nb_clf.predict(tf_idf_text)
    rf_prediction = rf_clf.predict(tf_idf_text)
    nb_prediction = "Male" if nb_prediction[0] == "m" else "Female"
    rf_prediction = "Male" if rf_prediction[0] == "m" else "Female"
    st.write("Prediction By Naive Bayes:", nb_prediction);
    st.write("Prediction By Random Forest:", rf_prediction);