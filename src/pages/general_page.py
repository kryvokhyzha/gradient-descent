import streamlit as st
from helper import markdown_to_string

def general_page():
    st.markdown(markdown_to_string('data/markdown/general_page.md'))
