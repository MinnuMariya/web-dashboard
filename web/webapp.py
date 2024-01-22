import streamlit as st
import torch
import json
from langchain.llms import GooglePalm
from dotenv import load_dotenv
import os
from icecream import ic

from langchain_utils.main import run_pipeline

load_dotenv()
llm = GooglePalm(model="gemini",temperature=0)

st.title("Detect microtargeting in tweets")

def predict_microtargeting(tweet:str):
    return run_pipeline(tweet,llm)

# Text input for the user to enter a tweet
tweet_text = st.text_area("Enter your tweet text here", "")

# Prediction button
if st.button("Predict"):
    if tweet_text:
        # Call your prediction function here, passing tweet_text
        result = predict_microtargeting(tweet_text)
        st.write(result)
    else:
        st.warning("Please enter a tweet text.")



