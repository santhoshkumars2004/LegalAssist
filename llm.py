import os 
from apikey import apikey
from click import prompt
from langchain import hub
from numpy import argsort
from pydantic_core import ArgsKwargs 
import streamlit as st
import pandas as pd 
import OpenAI

from langchain.llms import opena



#main

st.title('Legal Assitant')
st.header("find the ipc and SOP process")
st.subheader("solution")
#Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked={1:False}

def clicked(button):
    st.session_state.clicked[button]=True
st.button("Lets get started", on_click=clicked, args=[1])
if st.button('lets get started'):
    st.header("IPC PART1")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file.seek[0]

