import streamlit as st

# Set up individual pages for each homework
hw_01_page = st.Page("lab-01.py", title="Lab-01")
hw_02_page = st.Page("lab-02.py", title="Lab-02")
hw_03_page = st.Page("lab-03.py", title="Lab-03")
hw_04_page = st.Page("lab-04.py", title="Lab-04")
hw_05_page = st.Page("lab-05.py", title="Lab-05")
hw_06_page = st.Page("lab-06.py", title="Lab-06")
hw_07_page = st.Page("lab-07.py", title="Lab-07")
hw_08_page = st.Page("lab-08.py", title="Lab-08")
hw_09_page = st.Page("lab-09.py", title="Lab-09")
hw_10_page = st.Page("lab-10.py", title="Lab-10")
hw_11_page = st.Page("lab-11.py", title="Lab-11")
hw_12_page = st.Page("lab-12.py", title="Lab-12")

# Navigation setup with all homework pages
pg = st.navigation([
    hw_01_page, hw_02_page, hw_03_page, hw_04_page, hw_05_page,
    hw_06_page, hw_07_page, hw_08_page, hw_09_page, hw_10_page,
    hw_11_page, hw_12_page
])

# Configuration of the main app
st.set_page_config(page_title="HW Manager")

# Running the page navigation
pg.run()