import streamlit as st

from rajastan.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;font-size: 30px;'>Rajasthan Police GPT</h1>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("About")
        st.markdown(
            " The Rajasthan Police, established in 1951, is the law enforcement agency of the state, headquartered in Jaipur. Led by the Director General of Police, it is organized into units such as CID, Traffic Police, and specialized divisions. The force is responsible for maintaining law and order, preventing and investigating crimes, and managing traffic.Actively engaging in community policing, the Rajasthan Police emphasizes training, technological integration, and collaboration with other agencies to ensure effective law enforcement and the safety of the residents."
        )
        st.markdown(" YOU'RE SECURITY MATTERS!"
        )
        
