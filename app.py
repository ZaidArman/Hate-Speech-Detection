# import streamlit as st
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from scipy.special import softmax

# # Hate Speech Detection Model
# hate_speech_model_name = "austinmw/distilbert-base-uncased-finetuned-tweets-sentiment"
# hate_speech_tokenizer = AutoTokenizer.from_pretrained(hate_speech_model_name)
# hate_speech_model = AutoModelForSequenceClassification.from_pretrained(hate_speech_model_name)

# # Ensure the model is on the right device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hate_speech_model.to(device)

# # Preprocess text for Hate Speech Model
# def preprocess(text):
#     new_text = []  # replace usernames and URLs - if using tweets data
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

# # Predict sentiment for Hate Speech Model
# def predict_sentiment(text):
#     text = preprocess(text)
#     encoded_input = hate_speech_tokenizer(text, return_tensors='pt').to(device)
#     with torch.no_grad():
#         output = hate_speech_model(**encoded_input)
#     scores = output.logits[0].cpu().numpy()
#     scores = softmax(scores)
#     labels = ['Hate', 'neutral', 'Not Hate']
#     result_label = labels[scores.argmax()]
#     return result_label

# # Offensive Language Detection Model
# def load_offensive_model(token):
#     model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/OffensiveSpeechDetector", token=token)
#     tokenizer = AutoTokenizer.from_pretrained("KoalaAI/OffensiveSpeechDetector", token=token)
#     return model, tokenizer

# def predict_offensive(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_id = torch.argmax(logits, dim=-1).item()
#     predicted_label = model.config.id2label[predicted_class_id]
#     return predicted_label

# # Streamlit App
# def main():
#     st.title("Text Analysis App")
#     st.write("Enter a sentence to get its prediction:")

#     task = st.selectbox("Choose the model for prediction:", ["Hate Speech Detection", "Offensive Language Detection"])
#     user_input = st.text_area("Enter sentence", "")

#     if st.button("Predict"):
#         if user_input:
#             if task == "Hate Speech Detection":
#                 result = predict_sentiment(user_input)
#                 st.write(f"Hate Speech Prediction: {result}")
#             else:
#                 token = "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"
#                 offensive_model, offensive_tokenizer = load_offensive_model(token)
#                 result = predict_offensive(user_input, offensive_model, offensive_tokenizer)
#                 st.write(f"Offensive Language Prediction: {result}")
#         else:
#             st.write("Please enter a sentence to analyze")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

# Hate Speech Detection Model
hate_speech_model_name = "austinmw/distilbert-base-uncased-finetuned-tweets-sentiment"
hate_speech_tokenizer = AutoTokenizer.from_pretrained(hate_speech_model_name)
hate_speech_model = AutoModelForSequenceClassification.from_pretrained(hate_speech_model_name)

# Ensure the model is on the right device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hate_speech_model.to(device)

# Preprocess text for Hate Speech Model
def preprocess(text):
    new_text = []  # replace usernames and URLs - if using tweets data
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Predict sentiment for Hate Speech Model
def predict_sentiment(text):
    text = preprocess(text)
    encoded_input = hate_speech_tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = hate_speech_model(**encoded_input)
    scores = output.logits[0].cpu().numpy()
    scores = softmax(scores)
    labels = ['Hate', 'neutral', 'Not Hate']
    result_label = labels[scores.argmax()]
    return result_label

# Offensive Language Detection Model
def load_offensive_model(token):
    model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/OffensiveSpeechDetector", token=token)
    tokenizer = AutoTokenizer.from_pretrained("KoalaAI/OffensiveSpeechDetector", token=token)
    return model, tokenizer

def predict_offensive(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    return predicted_label


def main():
    st.title(" \t Hate Speech Detection App ")

    # task = st.sidebar.selectbox("Choose the model for prediction:", ["Hate Speech Detection", "Offensive Language Detection"])
    user_input = st.text_area("Enter sentence or Paragraph \n ", "")

    if st.button("Predict"):
        if user_input:
            # if task == "Hate Speech Detection":
            result = predict_sentiment(user_input)
            st.write(f":blue[Prediction]:  {result}")
        # else:
        #     token = "Your HuggingFace token"
        #     offensive_model, offensive_tokenizer = load_offensive_model(token)
        #     result = predict_offensive(user_input, offensive_model, offensive_tokenizer)
        #     st.write(f":blue[Prediction]:  {result}")
        else:
            st.write("Please enter a sentence to analyze")

if __name__ == "__main__":
    main()
