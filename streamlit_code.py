import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Crystal's Final Project (in Process): Health Article Classification by Importance",
    layout="wide"
)

st.title("Crystal's Final Project (in Process): Health Article Classification by Importance")
st.markdown("""
The dataset contains all pageviews associated with health articles (n=3177) that are either labeled as high, medium, or low level importance.
The scope of the dataset is limited to the U.S. in the English language from 2023 to 2024.  

The goal of this research project is to 1) classify articles by the level of importance and 2) test if there is a difference between pageviews across time & across level of importance.       
""")

st.header("1. Assembling Full Dataset")
st.markdown("There are 6 subcategories: high, medium, low, top importance & NA/unkown importance.")
st.markdown("Each category is associated with a talk page. Using the talk page, I then accessed the page properties via mwClient API calls. I finally talked to the duckdb server and filtered the full dataset using this unique QID list.")
st.markdown("In the future, I will try to reassemble this dataset to also retrieve other properties like page size and number of unique revisions, as these are helpful features that can be used to build a classifier.")

@st.cache_data
def load_data_1():
    try:
        # Load the uploaded file
        df1 = pd.read_csv("unique_health_articles.csv")
        
        # Remove "Unnamed: 0" column if it exists (common artifact in CSVs)
        if "Unnamed: 0" in df1.columns:
            df1 = df1.drop(columns=["Unnamed: 0"])
            
        # Ensure correct data types
        df1['total_pageviews'] = pd.to_numeric(df['total_pageviews'], errors='coerce').fillna(0)
        df1['description'] = df1['description'].fillna("")
        
        return df1
    except FileNotFoundError:
        st.error("File 'unique_health_articles.csv' not found. Please upload it.")
        return pd.DataFrame()

df1 = load_data_1()
st.write("Unique health articles - raw data overview:", df1.head())
st.metric("Total Articles", f"{len(df1)}")
st.metric("Total Pageviews", f"{df1['total_pageviews'].sum():,}")
st.metric("Average Pageviews / Article", f"{df1['total_pageviews'].mean():,.0f}")


st.subheader("Pageview Across Category Analysis")
cat_counts = df1['category'].value_counts().rename_axis('Category').reset_index(name='Count')
    
fig_bar = px.bar(
    cat_counts, 
    x='Category', 
    y='Count', 
    color='Category',
    text='Count',
    title="Number of Articles per Category",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_bar.update_traces(textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True)


def load_data_2():
    try:
        # Load the uploaded file
        df2 = pd.read_csv("all_health_articles.csv")
        
        return df2
    except FileNotFoundError:
        st.error("File not found. Please upload it.")
        return pd.DataFrame()

st.subheader("Pageview Across Time Analysis")

