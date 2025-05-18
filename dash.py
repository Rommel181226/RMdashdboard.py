import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import requests

# Token and GitHub details
#token = "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"
#repo_owner = "romero220"
#repo_name = "projectmanagement"
#branch = "main"

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit config
st.set_page_config(page_title="Task Dashboard", layout="wide")

# Color palette - changed to tab10 for better category distinction
# We'll dynamically set number of colors needed based on unique users
def get_color_palette(num_colors):
    return sns.color_palette("tab10", n_colors=num_colors).as_hex()

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)
    combined_df['Hours'] = combined_df['minutes'] / 60

    combined_df['task_wo_punct'] = combined_df['task'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].apply(lambda x: re.split(r'\W+', str(x).lower()))
    stopword = nltk.corpus.stopwords.words('english')
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(lambda x: [word for word in x if word not in stopword])
    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return combined_df

# Load data
combined_df = load_data()

# Compute keyword_counts after loading data so it's available globally
if not combined_df.empty:
    keyword_counts = pd.Series(
        {kw: combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: kw in x).sum()
         for kw in set([item for sublist in combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] for item in sublist])}
    )
    keyword_counts = keyword_counts.sort_values(ascending=False).reset_index()
    keyword_counts.columns = ['keyword', 'count']
    # Drop rows where 'keyword' contains only numbers
    keyword_counts = keyword_counts[~keyword_counts['keyword'].str.fullmatch(r'\d+')]
    keyword_counts['keyword+count'] = keyword_counts['keyword'] + " (" + keyword_counts['count'].astype(str) + ")"
else:
    keyword_counts = pd.DataFrame(columns=['keyword', 'count', 'keyword+count'])

# Sidebar filters
st.sidebar.header("Filters")

# Precompute unique values for faster UI rendering
project_ids = combined_df['ProjectID'].dropna().unique()
full_names = combined_df['Full_Name'].dropna().unique()
keyword_options = keyword_counts['keyword+count'].tolist()

ProjectID = st.sidebar.multiselect("Select Project ID", options=project_ids)
keyword_finder = st.sidebar.multiselect("Select a Keyword", options=keyword_options)
date_filter = st.sidebar.date_input("Filter by Date", [])
search_term = st.sidebar.text_input("Search Task", "")
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=full_names)
time_group = st.sidebar.selectbox("Group by Time Period", options=["Yearly", "Monthly", "Weekly", "Daily"])

# Prepare keyword lookup for filtering
keyword_lookup = {f"{row['keyword']} ({row['count']})": row['keyword'] for _, row in keyword_counts.iterrows()}

# Filter data efficiently
filtered_data = combined_df

if ProjectID:
    filtered_data = filtered_data[filtered_data['ProjectID'].isin(ProjectID)]

if keyword_finder:
    selected_keywords = [keyword_lookup[k] for k in keyword_finder]
    filtered_data = filtered_data[
        filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
            lambda x: any(word in x for word in selected_keywords)
        )
    ]

if len(date_filter) == 2:
    filtered_data = filtered_data.copy()
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[
        (filtered_data["started_at"] >= start_date) & (filtered_data["started_at"] <= end_date)
    ]

if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]

if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

filtered_data = filtered_data.reset_index(drop=True)

# Download filtered data button
csv_data = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="📥 Download Filtered CSV", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

# File upload and GitHub push
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    file_content = uploaded_file.read()
    file_name = uploaded_file.name
    encoded_content = base64.b64encode(file_content).decode()
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    data = {"message": f"Adding {file_name} via Streamlit", "content": encoded_content, "branch": branch}

    if st.sidebar.button("Confirm Upload"):
        response = requests.put(url, headers=headers, json=data)
        if response.status_code == 201:
            st.sidebar.success(f"File '{file_name}' uploaded!")
        else:
            st.sidebar.error(f"Upload failed: {response.json().get('message', 'Unknown error')}")

# Tabs
tab1, = st.tabs(["Overview"])

with tab1:
    st.header("Overview of Data Files")

    def get_file_details():
        files = []
        for file in [f for f in os.listdir('.') if f.endswith('.csv')]:
            try:
                df = pd.read_csv(file)
                files.append({'Filename': file, 'Rows (Excluding Headers)': len(df)})
            except Exception as e:
                st.warning(f"Error reading {file}: {e}")
        return files

    details = get_file_details()
    st.subheader("Uploaded CSV Files")
    st.table(pd.DataFrame(details if details else [{"Filename": "", "Rows (Excluding Headers)": 0}]))

    st.subheader("Preview of Filtered Data (First 100 Rows)")
    st.dataframe(filtered_data.head(100), use_container_width=True)

    st.subheader("Missing Values by Column")
    missing_counts = filtered_data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if not missing_counts.empty:
        missing_df = pd.DataFrame({'Column': missing_counts.index, 'Missing Values': missing_counts.values})
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Missing Values',
            color='Missing Values',
            color_continuous_scale='Reds',
            title="Number of Missing Values per Column",
            labels={'Missing Values': 'Count of NaNs'},
            hover_data={'Column': True}
        )
        fig_missing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values in the filtered dataset.")

    # Prepare for time grouping bar chart
    st.subheader(f"Task Count by User - {time_group} View")

# Convert started_at to datetime once more if needed
filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce")

# Time grouping based on selection
if time_group == "Yearly":
    filtered_data["TimeGroup"] = filtered_data["started_at"].dt.year.astype(str)
elif time_group == "Monthly":
    filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%Y-%m')
elif time_group == "Weekly":
    filtered_data["TimeGroup"] = filtered_data["started_at"].dt.to_period("W").astype(str)
elif time_group == "Daily":
    filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%Y-%m-%d')

# Group data
grouped = filtered_data.groupby(["Full_Name", "TimeGroup"])['Hours'].sum().reset_index()

# Sort time groups chronologically (handles weekly format)
grouped["TimeGroupSort"] = pd.to_datetime(grouped["TimeGroup"].str.split("/").str[0], errors='coerce')
grouped = grouped.sort_values("TimeGroupSort")

# Prepare color palette for the number of unique users in filtered data
unique_users = grouped['Full_Name'].nunique()
color_palette = get_color_palette(unique_users)

# Plotly bar chart
fig_timegroup = px.bar(
    grouped,
    x="TimeGroup",
    y="Hours",
    color="Full_Name",
    barmode="group",
    title=f"Accumulated Hours per User by {time_group}",
    labels={"TimeGroup": "Time", "Hours": "Total Hours"},
    color_discrete_sequence=color_palette,
    height=500
)

# Customize x-axis ticks
if time_group == "Yearly":
    fig_timegroup.update_xaxes(type="category", tickmode='linear', tickformat='%Y')
else:
    fig_timegroup.update_xaxes(type="category", tickangle=-45)

# Display plot
st.plotly_chart(fig_timegroup, use_container_width=True)
