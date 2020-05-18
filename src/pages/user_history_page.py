import streamlit as st
import pandas as pd
from db import db_clean, db_select


def user_history_page():
    st.title('User history')
    rows = db_select()

    col = ('Modification','Excecution time', 'Insert date', 'Hypothesis', 'Cost function', 'Regularization','Scalling function',
    'Reg coef', 'Learning rate', 'Early stopping', 'Number of itterations', 'Weights')

    df = pd.DataFrame(
    rows,
    columns=col)

    columns = st.multiselect(
    label='What information do you want to display?', options=df.columns)

    if columns:
        st.table(df[columns])