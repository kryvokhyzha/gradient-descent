import streamlit as st
import pandas as pd
from db.db import *


def user_history_page():
    st.title('User history')

    if st.sidebar.button("Delete history"):
        db_clean()

    rows = db_select()

    col = ('Modification','Excecution time', 'Insert date', 'Hypothesis', 'Cost function', 'Regularization','Scalling function',
    'Reg coef', 'Learning rate', 'Early stopping', 'Number of itterations', 'Weights')

    df = pd.DataFrame(rows, columns=col)

    columns = st.multiselect(label='What information do you want to display?', options=col, default=list(col))

    if columns and df.shape[0]:
        st.write(df[columns])
    else:
        st.warning('User history is empty')
                      