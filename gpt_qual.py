import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import logging


# Streamlit app
st.title('GPT Qual')

# API Key input
api_key = st.text_input("Enter your OpenAI API key", type="password")

# Configure logging to file
logging.basicConfig(filename='gpt_qual.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize a dictionary to store categories
categories_dict = {}

def analyze_sentiment(text):
    prompt = f"Analyze the sentiment of the following text and respond with a JSON object containing 'sentiment' with values 'Positive', 'Negative', or 'Neutral': {text}. "
    return call_openai_api(prompt).get('sentiment')

def analyze_categorize(text, categories_dict):
    existing_categories = ', '.join(categories_dict.keys())
    prompt = f"Categorize the following text: {text} based on its essential meaning. Essentially a 1-5 word summary. Respond with one of these existing categories: {existing_categories} or make a new one. Respond with a JSON object containing 'category'."
    category = call_openai_api(prompt).get('category')
    if category:
        if category not in categories_dict:
            categories_dict[category] = 1
        else:
            categories_dict[category] += 1
    return category

def analyze_mark_salient(text):
    prompt = f"Determine if the following text is salient for marketing purposes: {text}. Respond with a JSON object containing 'salient' with values 'True' or 'False'."
    return call_openai_api(prompt).get('salient')

def analyze_custom(text, custom_prompt):
    prompt = f"{custom_prompt}: {text}. Respond with a JSON object containing 'response'."
    return call_openai_api(prompt).get('response')

def call_openai_api(prompt):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        max_tokens=100,
        messages=[
            {"role": "system", "content": "You are an AI model that provides structured JSON responses."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        logging.info(f'Completed OpenAI API Call with input characters: {len(prompt)} and output characters: {len(response.choices[0].message.content)}')
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from API"}

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file and api_key:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame:", df)

    # Column selector
    column = st.selectbox("Select the column to analyze", df.columns)

    # Analysis type selector
    analysis_type = st.selectbox("Select the type of analysis", ['sentiment', 'categorize', 'mark salient', 'custom'])

    if analysis_type == 'custom':
        custom_prompt = st.text_input("Enter your custom prompt:")

    # Output column name
    output_column_name = st.text_input("Enter the name for the output column", "analysis_result")

    if st.button("Preview Analysis for First 10 Rows"):
        # Progress indicator
        preview_progress_text = st.empty()
        preview_progress_bar = st.progress(0)

        # Apply analysis to the first 10 rows of the selected column
        preview_results = []
        for i, text in enumerate(df[column].head(10)):
            limited_text = str(text)[:500]
            if analysis_type == 'sentiment':
                result = analyze_sentiment(limited_text)
            elif analysis_type == 'categorize':
                result = analyze_categorize(limited_text, categories_dict)
            elif analysis_type == 'mark salient':
                result = analyze_mark_salient(limited_text)
            elif analysis_type == 'custom':
                result = analyze_custom(limited_text, custom_prompt)
            preview_results.append(result)

            # Update progress
            preview_progress = (i + 1) / len(df.head(10)[column])
            preview_progress_bar.progress(preview_progress)
            preview_progress_text.text(f'Progress: {int(preview_progress * 100)}%')
        
        preview_df = df.head(10).copy()
        preview_df[output_column_name] = preview_results

        # Display the DataFrame with the analysis result for the first 10 rows
        st.write("Preview of the DataFrame with analysis (first 10 rows):")
        st.write(preview_df)

    if st.button("Analyze Entire DataFrame"):
        # Progress indicator
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Apply analysis to the selected column
        results = []
        for i, text in enumerate(df[column]):
            if analysis_type == 'sentiment':
                result = analyze_sentiment(text)
            elif analysis_type == 'categorize':
                result = analyze_categorize(text, categories_dict)
            elif analysis_type == 'mark salient':
                result = analyze_mark_salient(text)
            elif analysis_type == 'custom':
                result = analyze_custom(text, custom_prompt)
            results.append(result)

            # Update progress
            progress = (i + 1) / len(df[column])
            progress_bar.progress(progress)
            progress_text.text(f'Progress: {int(progress * 100)}%')

        df[output_column_name] = results

        # Display the DataFrame with the new column
        st.write("DataFrame with analysis:", df)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='analyzed_data.csv', mime='text/csv')

    # Display the categories
    if analysis_type == 'categorize':
        categories_df = pd.DataFrame([
            {
                "category": category,
                "count": categories_dict[category]
            }
            for category in categories_dict.keys()
        ])
        st.write("Categories:", categories_df)
else:
    st.write("Please upload a CSV file and enter your OpenAI API key.")