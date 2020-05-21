import streamlit as st
from helper import markdown_to_string

def about_us_page():
    st.markdown(markdown_to_string('data/markdown/about_us_page.md'))
