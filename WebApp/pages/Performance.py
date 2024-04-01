import streamlit as st
import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from SA_system import feedback_data_df

st.set_page_config(page_title="SAS", layout="wide")

# Fine-tune the model with feedback data
def fine_tune_model_with_feedback(feedback_data_df, batch_size=4):
    model_path = '../Model/advanced_model/'
    num_epochs = 3

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Prepare feedback data
    data_texts = feedback_data_df['text'].tolist()  
    data_labels = feedback_data_df['dominant_prediction'].tolist()

    inputs = tokenizer(data_texts, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(data_labels)

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fine-tune the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids, attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Optionally, log training progress
        average_loss = total_loss / len(dataloader)
        st.write(f"Epoch [ {epoch+1} / {num_epochs} ],  Average Loss: {average_loss}")

    # Save the fine-tuned model
    new_model_path = '../Model/advanced_model/'
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    return new_model_path

def evaluate_model():
    new_model_path = '../Model/advanced_model/'

    # Initialize lists to store labels, predictions, and confidence scores
    all_labels = []
    all_confidence_scores = []
    texts = []
    all_dom_predictions = []
    all_sub_predictions = []

    # Define batch size
    batch_size = 5000 

    # Load test dataset (skip the header)
    with open('./data/test_data.csv', 'r') as f:
        # Skip the header if present
        next(f)
        # Read the rest of the lines
        texts = [line.strip() for line in f]

    # Load test labels from the json file
    with open('./data/test_labels.json', 'r') as f:
        test_labels = json.load(f)

    # Instantiate tokenizer and DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained(new_model_path)
    finetuned_model = DistilBertForSequenceClassification.from_pretrained(new_model_path)
    finetuned_model.eval()  # Put the model in evaluation mode

    # Process data in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]

        # Preprocess the data
        tokenized_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)

        # Generate predictions
        with torch.no_grad():  # Avoid computing gradients during inference
            outputs = finetuned_model(**tokenized_inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Finding Dom and Sub predictions
        top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)
        dom_predictions = [int(pred) for pred in top2_indices[:, 0]]
        sub_predictions = [int(pred) for pred in top2_indices[:, 1]]

        # Store labels, predictions, and confidence scores for this batch
        all_labels.extend(batch_labels)
        all_dom_predictions.extend(dom_predictions)
        all_sub_predictions.extend(sub_predictions)
        all_confidence_scores.extend(top2_probs[:, 0].tolist())  # Confidence scores for Dom predictions

        # Free up memory by deleting variables that are no longer needed
        del tokenized_inputs, outputs

    # Determine if at least one correct sentiment is predicted
    correct_sentiment_predicted = any(label == dom_pred or label == sub_pred for label, dom_pred, sub_pred in zip(all_labels, all_dom_predictions, all_sub_predictions))

    # Calculate metrics across all predictions
    accuracy = accuracy_score(all_labels, all_dom_predictions) if correct_sentiment_predicted else accuracy_score(all_labels, all_sub_predictions)
    precision = precision_score(all_labels, all_dom_predictions, average="weighted") if correct_sentiment_predicted else precision_score(all_labels, all_sub_predictions, average="weighted")
    recall = recall_score(all_labels, all_dom_predictions, average="weighted") if correct_sentiment_predicted else recall_score(all_labels, all_sub_predictions, average="weighted")
    f1 = f1_score(all_labels, all_dom_predictions, average="weighted") if correct_sentiment_predicted else f1_score(all_labels, all_sub_predictions, average="weighted")

    # Calculating the confidence scores
    average_confidence_score = sum(all_confidence_scores) / len(all_confidence_scores)
    max_confidence_score = max(all_confidence_scores)
    min_confidence_score = min(all_confidence_scores)

    # Count positive and negative confidence scores
    positive_count = sum(1 for score in all_confidence_scores if score >= 0.6)
    negative_count = len(all_confidence_scores) - positive_count

    # 3.30 minutes

    return accuracy, precision, recall, f1, average_confidence_score, max_confidence_score, min_confidence_score, positive_count, negative_count

but_col1, but_col2 = st.columns([1, 1])

with but_col1:
    st.markdown("# Model Evaluation")
    st.markdown("Evaluate the current model's performance.")
    if st.button("Evaluate", key='eval_model_button'):
        with st.spinner("Evaluating the model..."):
            accuracy, precision, recall, f1, average_confidence_score, max_confidence_score, min_confidence_score, positive_count, negative_count = evaluate_model()
            # Printing the evaluation metrics
            st.markdown("### Evaluation Metrics")
            st.write(f"Accuracy: {round(accuracy * 100)}%")
            st.write(f"Precision: {round(precision * 100)}%")
            st.write(f"Recall: {round(recall * 100)}%")
            st.write(f"F1 Score: {round(f1 * 100)}%")
            st.write(" ")
            st.markdown("### Model's Prediction Confidence")
            st.write(f"Average Confidence Score: {round(average_confidence_score * 100)}%")
            st.write(f"Max Confidence Score: {round(max_confidence_score * 100)}%")
            st.write(f"Min Confidence Score: {round(min_confidence_score * 100)}%")
            st.write(" ")
            st.write(f"Positive Confidence Count: {positive_count}")
            st.write(f"Negative Confidence Count: {negative_count}")
        st.write("Model evaluation has been completed.")

with but_col2:
    st.markdown("# Self Learning Feature")
    st.markdown('Self learn using retrieved data from the API.')
    if st.button("Self Learn", key='self_learn_button'):
        with st.spinner("Fine-tuning the model..."):
            new_model_path = fine_tune_model_with_feedback(feedback_data_df)
        st.write("Model has been fine-tuned with feedback data.")


