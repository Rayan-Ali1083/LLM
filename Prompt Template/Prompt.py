from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()
HF_API = os.getenv('HF_API')

# Model Import

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    huggingfacehub_api_token=HF_API,
    temperature=0,
    timeout=180,
    max_new_tokens=25
)

model = ChatHuggingFace(llm=llm)


# Front end

st.header("Cats Informtion")

cats_input = st.selectbox("Select type of cat" , [
    "Persian",
    "Maine Coon",
    "Bengal",
    "Sphynx",
    "Ragdoll",
    "Scottish Fold",
    "Abyssinian",
    "British Shorthair",
    "Norwegian Forest Cat"
])
info_length = st.selectbox('Select length of information', [
    'three',
    'five',
    'ten'
])


# Prompt template

template = load_prompt('template.json')


# Invoke Answers based on template

if st.button('Give Info'):
    chain = template | model
    result = chain.invoke({
        'cats_input': cats_input,
        'info_length': info_length
    })
    st.write(result.content)
