# home.py
"""This module constructs a simple chatbot webpage."""

import streamlit as st
from tools.langchain_tools import classify_input
from tools.langchain_tools import get_response
from langchain_core.messages import HumanMessage, AIMessage


# Configs streamlit
st.set_page_config(page_title="Tour Guide AI Bot v1", page_icon="🤖")
st.title("Tour Guide AI Bot")

# Declares a chat_history variable to save the previous conversations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Max, your personal tour guide. Tell me an attraction to start the tour.")
    ]

# Gets user input
user_input = st.chat_input("Type your message here...")

# Validates user input
# If user input is valid, invokes methods to get response. Otherwise, does nothing.
if user_input and user_input != "":
    keyword = classify_input(user_input, st.session_state.chat_history)
    response = get_response(user_input, keyword)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

# Updates the webpage with chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
