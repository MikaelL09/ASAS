
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch
import re 
import nltk
from nltk.corpus import stopwords
from apify_client import ApifyClient
from collections import Counter


# Initialize the ApifyClient with your API token
client = ApifyClient("apify_api_K1KJtKoCp6dDVQLk6pYpAbocnfWfjv1JGH6e")

#, "lang:en f1", "lang:en formula one"

def run_api_actor():
    # Prepare the Actor input
    run_input = {
        "searchTerms": ["lang:en formula 1"],
        "searchMode": "live",
        "maxTweets": 1000,
        "maxRequestRetries": 5,
        "addUserInfo": False,
        "scrapeTweetReplies": False,
        "urls": ["https://twitter.com/search?q=lang%3Aen%20formula%201&src=recent_search_click"],
    }

    # Run the Actor and wait for it to finish
    api_run = client.actor("heLL6fUofdPgRXZie").call(run_input=run_input)
    # Return the run
    return api_run

def get_api_dataset(api_run):
    # Get the dataset from the run
    api_dataset = client.dataset(api_run["defaultDatasetId"]).get()
    # Fetch the 'text' field from each item in the dataset
    texts = [item['full_text'] for item in client.dataset(api_run["defaultDatasetId"]).iterate_items()]
    # Create a DataFrame with the 'text' data
    api_dataset = pd.DataFrame({'full_text': texts})
    # Return the dataset
    return api_dataset

api_dataset = get_api_dataset(run_api_actor())

# PREPROCESSING THE DATA
emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"
                          u"\U0001F300-\U0001F5FF"
                          u"\U0001F680-\U0001F6FF"
                          u"\U0001F1E0-\U0001F1FF"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def clean_and_preprocess(tweet):
    if not isinstance(tweet, str):
        return ""
    # Combine regular expressions
    temp = re.sub(r'http\S+|@\S+|#|\[.*?\]|[()!?]', ' ', tweet.lower())
    temp = re.sub("[^a-z0-9\s]", " ", temp)
    # Lemmatization (NLTK library) using list comprehension
    words = [lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', temp) if word not in stop_words]
    # Join the list of words back into a sentence
    processed_sentence = " ".join(words)
    # Return the processed data
    return processed_sentence

# Apply the clean_and_preprocess function to the "Text" column
prediction_data = api_dataset.copy()
prediction_data['text'] = prediction_data['full_text'].apply(clean_and_preprocess)
prediction_data = list(prediction_data['text'])

# Define the label-to-id mapping used during training
label2id = {'Admiration': 0, 'Amusement': 1, 'Anger': 2, 'Annoyance': 3, 'Approval': 4, 'Caring': 5, 'Confusion': 6,
            'Curiosity': 7, 'Desire': 8, 'Disappointment': 9, 'Disapproval': 10, 'Disgust': 11, 'Embarrassment': 12,
            'Excitement': 13, 'Fear': 14, 'Gratitude': 15, 'Grief': 16, 'Joy': 17, 'Love': 18, 'Nervousness': 19,
            'Optimism': 20, 'Pride': 21, 'Realization': 22, 'Relief': 23, 'Remorse': 24, 'Sadness': 25, 'Surprise': 26,
            'Neutral': 27}

# file_path = "./Dataset/loaded_f1_data.json"

# def load_and_sample_data(file_path, n=1000):
#     # Load the dataset
#     with open(file_path, "r") as file:
#         prediction_data = json.load(file)

#     # Convert data to DataFrame
#     df_api_data = pd.DataFrame(prediction_data, columns=['text'])

#     # Shuffle and sample
#     df_shuffled = df_api_data.sample(frac=1)
#     sampled_texts = df_shuffled['text'].sample(n=n).tolist()

#     return sampled_texts

# # Get the random sample of data
# random_data = load_and_sample_data(file_path=file_path, n=1000)

def predict_sentiment(prediction_data):
    #Loading custom model
    model_path = '../Model/final_model/'

    # Create a reverse mapping (id-to-label) for converting numerical labels to string labels
    id_to_label = {v: k for k, v in label2id.items()}

    # Instantiate tokenizer and DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    finetuned_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    finetuned_model.eval()  

    # Initialize lists to store results
    predicted_results = []
    sentiment_scores = []

    # Process each input sentence
    for input_sentence in prediction_data:
        # Tokenize the input
        tokenized_input = tokenizer(input_sentence, return_tensors="pt")

        # Generate predictions
        with torch.no_grad():  # Avoid computing gradients during inference
            outputs = finetuned_model(**tokenized_input)

        # Obtain predicted probabilities
        predicted_probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get the top two predictions and their probabilities
        top2_indices = torch.topk(predicted_probabilities, 2, dim=1)
        dom_prediction = id_to_label.get(top2_indices.indices[0][0].item(), "Unknown")
        sub_prediction = id_to_label.get(top2_indices.indices[0][1].item(), "Unknown")

        # Append the input sentence along with its dominant and sub predictions to the result list
        predicted_results.append((input_sentence, dom_prediction, sub_prediction))
        sentiment_scores.append(predicted_probabilities.tolist())

    # Calculate the highest sentiment prediction inside the loop
    average_sentiment_scores = torch.tensor(sentiment_scores).mean(dim=0)
    highest_average_sentiment_index = torch.argmax(average_sentiment_scores).item()
    highest_average_sentiment = id_to_label.get(highest_average_sentiment_index, "Unknown")

    # Create a DataFrame with tweets, dominant predictions, and sub predictions
    feedback_data_df = pd.DataFrame(predicted_results, columns=['text', 'dominant_prediction', 'sub_prediction'])

    return predicted_results, highest_average_sentiment,feedback_data_df

# Get predicted data and average sentiment
predicted_data, overal_sentiment,feedback_data = predict_sentiment(prediction_data)

tweets = []
dominant_predictions = []
sub_predictions = []

for tweet, dom_pred, sub_pred in predicted_data:
    tweets.append(tweet)
    dominant_predictions.append(dom_pred)
    sub_predictions.append(sub_pred)

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
sentiment_cat_counts = Counter(sentiment_category[sentiment] for sentiment in dominant_predictions)


feedback_data_df = feedback_data.copy()
# Map the string labels to their corresponding numeric labels
feedback_data_df['dominant_prediction'] = feedback_data_df['dominant_prediction'].map(label2id)
# Drop sub prediction column
feedback_data_df = feedback_data_df.drop(columns=['sub_prediction'])

