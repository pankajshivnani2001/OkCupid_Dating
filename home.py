import streamlit as st
import dating


pages = {
            "Understanding the Dataset" : dating
        }

option = st.sidebar.selectbox(
    "What would you like to open?",
    ("Understanding the Dataset")
)


page = pages[option]
page.app()