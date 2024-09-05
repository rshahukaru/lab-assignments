import streamlit as st

lab_01_page = st.Page("lab-01.py", title="Lab 01")
lab_02_page = st.Page("lab-02.py", title="Lab 02", default = True)

pg = st.navigation([lab_01_page, lab_02_page])
st.set_page_config(page_title="Multi Page Streamlit App")
pg.run()