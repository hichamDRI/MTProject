from subprocess import Popen
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

def download_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
    return model, tokenizer

model_name_fr_ar="Helsinki-NLP/opus-mt-fr-ar"
model_fr_ar, tokenizer_fr_ar = download_model(model_name_fr_ar)
model_name_ar_fr="Helsinki-NLP/opus-mt-ar-fr"
model_ar_fr, tokenizer_ar_fr = download_model(model_name_ar_fr)

def load_jupyter_server_extension(nbapp):
    """serve the streamlit app"""
    Popen(
        [
            "streamlit", 
            "run", 
            "app.py", 
            "--browser.serverAddress=0.0.0.0", 
            "--server.enableCORS=False",
            "--runner.fixMatplotlib=True"
        ]
    )

