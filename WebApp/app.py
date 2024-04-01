import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from SA_system import label2id, tweets, dominant_predictions, sub_predictions, overal_sentiment, predicted_data, predict_sentiment, sentiment_cat_counts
from SA_system import run_api_actor, get_api_dataset, clean_and_preprocess, api_dataset


# Set the page configuration
st.set_page_config(page_title="SAS", layout="wide")

# Set page title
st.title("Sentiment Analysis Dashboards & Individual Insights")

st.markdown('Retrieve new data from API.')

if st.button("Get New Data", key='get_data_button'):

    with st.spinner("Gathering the Data..."):
        # Run the API and get the dataset
        api_dataset = get_api_dataset(run_api_actor())
        #Apply the clean_and_preprocess function to the "Text" column
        prediction_data = api_dataset.copy()
        prediction_data['text'] = prediction_data['full_text'].apply(clean_and_preprocess)
        prediction_data = list(prediction_data['text'])
        # # Get the random sample of data
        # random_data = load_and_sample_data(file_path=file_path, n=1000)
        # Get predicted data and average sentiment
        predicted_data, overal_sentiment, feedback_data = predict_sentiment(prediction_data)
        tweets = []
        dominant_predictions = []
        sub_predictions = []

        for tweet, dom_pred, sub_pred in predicted_data:
            tweets.append(tweet)
            dominant_predictions.append(dom_pred)
            sub_predictions.append(sub_pred)

        # Reset the sentiment counts
        sentiment_cat_counts.clear()

        # Recategorize sentiments into Positive, Negative, and Neutral categories
        Positive_label = ['Admiration', 'Amusement', 'Approval', 'Caring', 'Desire', 'Excitement', 'Gratitude', 'Joy', 'Love', 'Optimism', 'Pride', 'Relief']
        Negative_label = ['Anger', 'Annoyance', 'Disappointment', 'Disapproval', 'Disgust', 'Embarrassment', 'Fear', 'Grief', 'Nervousness', 'Remorse', 'Sadness']
        Neutral_label = ['Surprise', 'Neutral', 'Confusion', 'Curiosity', 'Realization']

        # Map the sentiments to their respective categories
        sentiment_category = {}
        for sentiment in Positive_label:
            sentiment_category[sentiment] = 'Positive'
        for sentiment in Negative_label:
            sentiment_category[sentiment] = 'Negative'
        for sentiment in Neutral_label:
            sentiment_category[sentiment] = 'Neutral'

        # Count occurrences of each sentiment category
        sentiment_cat_counts.update(sentiment_category[sentiment] for sentiment in dominant_predictions)

        feedback_data_df = feedback_data.copy()
        # Map the string labels to their corresponding numeric labels
        feedback_data_df['dominant_prediction'] = feedback_data_df['dominant_prediction'].map(label2id)
        # Drop sub prediction column
        feedback_data_df = feedback_data_df.drop(columns=['sub_prediction'])

        api_run = True

api_run = False

st.markdown('---')
main_col1, main_col2 = st.columns([1, 1])

# Define page content
with main_col1:

    # Update data and visualizations only if API has been run
    if api_run:
        # Count occurrences of each sentiment label
        sentiment_counts = {label: dominant_predictions.count(label) for label in label2id.keys()}

        # Sort sentiment counts by their values
        sorted_sentiment_counts = dict(sorted(sentiment_counts.items(), key=lambda item: item[1]))

        # Extract labels and counts for plotting
        labels = list(sorted_sentiment_counts.keys())
        counts = list(sorted_sentiment_counts.values())

        all_predictions = dominant_predictions + sub_predictions

        # Plot the bar chart
        st.markdown("<h2 style='text-align: center; color: white;'>Bar Chart Distribution</h2>", unsafe_allow_html=True)
        st.bar_chart(Counter(all_predictions))

        st.markdown('---')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Record Count")
            st.write(f"Total Records: {len(tweets)}")

        with col2:
            # Print highest average sentiment prediction
            st.subheader("Overall Sentiment")
            st.write(overal_sentiment)

        api_run = False  # Reset flag to re-running the API

    else:
        # Count occurrences of each sentiment label
        sentiment_counts = {label: dominant_predictions.count(label) for label in label2id.keys()}

        # Sort sentiment counts by their values
        sorted_sentiment_counts = dict(sorted(sentiment_counts.items(), key=lambda item: item[1]))

        # Extract labels and counts for plotting
        labels = list(sorted_sentiment_counts.keys())
        counts = list(sorted_sentiment_counts.values())

        all_predictions = dominant_predictions + sub_predictions

        # Plot the bar chart
        st.markdown("<h2 style='text-align: center; color: white;'>Bar Chart Distribution</h2>", unsafe_allow_html=True)
        st.bar_chart(Counter(all_predictions))
        
        st.markdown('---')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Record Count")
            st.write(f"Total Records: {len(tweets)}")

        with col2:
            # Print highest average sentiment prediction
            st.subheader("Overall Sentiment")
            st.write(overal_sentiment)

        api_run = False  # Reset flag to re-running the API

with main_col2:
    # Plot a pie chart
    st.markdown("<h2 style='text-align: center; color: white;'>Pie Chart Distribution</h2>", unsafe_allow_html=True)
    plot_pie = px.pie(values=sentiment_cat_counts.values(), names=sentiment_cat_counts.keys(), color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(plot_pie)

st.markdown('---')
st.title('Single Predictions')

# Set the number of records to display per page
records_per_page = 10

# Calculate the total number of pages
total_pages = len(predicted_data) // records_per_page + (1 if len(predicted_data) % records_per_page > 0 else 0)

# Select the current page
page_number = st.number_input("Page Number", min_value=1, max_value=total_pages, value=1, key='page_number')

# Calculate the start and end indices of records to display on the selected page
start_index = (page_number - 1) * records_per_page
end_index = min(start_index + records_per_page, len(predicted_data))

# Slice the data for the selected page
page_data = predicted_data[start_index:end_index]

# Create a DataFrame from the sliced data
table_data = pd.DataFrame(page_data, columns=['Text', 'Dominant Prediction', 'Sub Prediction'], index=range(start_index + 1, end_index + 1))

# Display the DataFrame as a table
st.table(table_data)