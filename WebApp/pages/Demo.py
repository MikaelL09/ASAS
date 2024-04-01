import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="SAS", layout="wide")

st.title("Demo Sentiment Analysis")

def pred_sentiment(input_sentence):
    # Define the label-to-id mapping used during training
    label2id = {'admiration': 0, 'amusement': 1, 'anger': 2, 'annoyance': 3, 'approval': 4, 'caring': 5, 'confusion': 6,
                'curiosity': 7, 'desire': 8, 'disappointment': 9, 'disapproval': 10, 'disgust': 11, 'embarrassment': 12,
                'excitement': 13, 'fear': 14, 'gratitude': 15, 'grief': 16, 'joy': 17, 'love': 18, 'nervousness': 19,
                'optimism': 20, 'pride': 21, 'realization': 22, 'relief': 23, 'remorse': 24, 'sadness': 25, 'surprise': 26,
                'neutral': 27}

    # Create a reverse mapping (id-to-label) for converting numerical labels to string labels
    id_to_label = {v: k for k, v in label2id.items()}

    # Instantiate tokenizer and DistilBERT model
    model_path = '../Model/final_model/'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    finetuned_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    finetuned_model.eval()  # Put the model in evaluation mode

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

    return dom_prediction, sub_prediction

input_sentence = st.text_input("Enter a sentence:")
if st.button("Predict"):
    dom_prediction, sub_prediction = pred_sentiment(input_sentence)
    st.markdown("---")
    st.write("Dominant Prediction:", dom_prediction)
    st.write("Sub Prediction:", sub_prediction)
