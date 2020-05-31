import streamlit as st
from pages import (general_page, gd_solution_page,
                   user_history_page, about_us_page,
                   comparison_page)
from db import db_creation


if __name__ == '__main__':
    db_creation()
    options = {'General': general_page,
               'Gradient Descent Solution': gd_solution_page,
               'Algorithms Comparison': comparison_page,
               'User History': user_history_page,
               'About Us': about_us_page
               }

    st.sidebar.header("Please, choose page")
    page = st.sidebar.radio('', key='page_choice_radio', options=list(options.keys()))
    
    options[page]()
