import streamlit as st
import dating, classifiers


pages = {
            "Understanding the Dataset" : dating,
            "Predicting Gender using Self Summary Essay" : classifiers
        }

option = st.sidebar.selectbox(
    "What would you like to open?",
    ("Understanding the Dataset", "Predicting Gender using Self Summary Essay")
)


page = pages[option]
page.app()