from operator import index
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

import pandas_profiling
import pandas as pd
import os 
import streamlit as st

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, index_col=None)
    return df

def explore_data(df):
    profile_df = df.profile_report()
    return profile_df

def run_model(df, chosen_target):
    setup(df, target=chosen_target)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model, 'best_model')

    with open('best_model.pkl', 'rb') as f: 
        model_bytes = f.read()
    return setup_df, compare_df, model_bytes

def main():
    image = Image.open('temp/wall-e.png')
    st.set_page_config(page_title='Auto ML Classifier', page_icon=image, layout="centered", initial_sidebar_state="auto", menu_items=None)

    if os.path.exists('./dataset.csv'): 
        df = pd.read_csv('dataset.csv', index_col=None)
    else:
        df = pd.DataFrame()

    with st.sidebar:
        st.image(image)
        st.title("Auto ML Classifier")
        choice = st.radio("Process the data here!", ["Upload","Profiling","Modelling"])
        st.info("This project application helps you build model and explore your data.")
        st.caption("This project is developed by [Bastian Armananta](https://www.linkedin.com/in/bastian-armananta/)")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file: 
            df = load_data(file)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if df.empty:
        st.warning("Please upload a dataset first")

    if choice == "Profiling" and not df.empty:
        st.title("Exploratory Data Analysis")
        profile_df = explore_data(df)
        st_profile_report(profile_df)

    if choice == "Modelling" and not df.empty:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'): 
            with st.spinner('Training Model...'):
                setup_df, compare_df, model_bytes = run_model(df, chosen_target)
            st.dataframe(setup_df)
            st.dataframe(compare_df)
            st.download_button('Download Model', model_bytes, file_name="best_model.pkl")

    if st.button("Clear Data"):
        df = pd.DataFrame()
        if os.path.exists("dataset.csv"):
            os.remove("dataset.csv")
            st.success("Data has been cleared!")
        else:
            st.warning("No data found to clear!")

if __name__ == '__main__':
    main()