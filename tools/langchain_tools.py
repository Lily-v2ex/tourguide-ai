# langchain_tools.py
"""This module provides custom tools that employ langchain technologies.
Methods:
    classify_input -- Classify the user input and return a pre-defined classification keyword.
    get_response -- Generate a response and return it in the form of string literal.
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import StructuredTool
from constants import ATTRACTION_SYSTEM_MESSAGE
from constants import OTHER_SYSTEM_MESSAGE


def set_api_key(keyword: str):
    """Store an API key as an environment variable"""
    if st.secrets[keyword] and st.secrets[keyword] != "":
        os.environ[keyword] = st.secrets[keyword]


# Sets API keys
set_api_key("OPENAI_API_KEY")
set_api_key("SERPAPI_API_KEY")

# Defines system messages for each keyword
msg_dict = {
    "Attraction": ATTRACTION_SYSTEM_MESSAGE,
    "Other": OTHER_SYSTEM_MESSAGE
}


def classify_input(user_input: str, chat_history: list) -> str:
    """Classify the user input and return a pre-defined classification.
    Keyword arguments:
        user_input -- the user input from the input field
        chat_history -- the chat history of the previous conversations
    """
    # Instantiates a model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    # Creates a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user input "
                   "which might reference context in the chat history, "
                   "Classify the latest user input into the following groups: "
                   "'Attraction' and 'Other' by the below rules."
                   "If the latest user input is the very first famous attraction"
                   "mentioned in the chat history "
                   "or it is a new attraction that was not in the same location "
                   "as the last attraction mentioned in the chat history, "
                   "then return 'Attraction'. "
                   "Else return 'Other'"
                   "Do not return an empty answer."
                   "EXAMPLE CONVERSATION"
                   "[Customer]: Qingjing Mosque"
                   "EXAMPLE OUTPUT"
                   "Attraction"
                   "EXAMPLE CONVERSATION"
                   "[User]: Qingjing Mosque"
                   "[User]: the Mingshan Hall"
                   "[User]: Forbidden city"
                   "EXAMPLE OUTPUT"
                   "Attraction"
                   "EXAMPLE CONVERSATION"
                   "[User]: Qingjing Mosque"
                   "[User]: the Mingshan Hall"
                   "[User]: Forbidden city"
                   "[User]: when to open"
                   "EXAMPLE OUTPUT"
                   "Other"
         ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    # Creates a smple chain
    chain = prompt | llm
    # Invokes the chain
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    # Prints the result in terminal
    print("-----Get response-----")
    print(response.content)
    return response.content


def get_response(user_input: str, keyword: str) -> str:
    """Generate a response and return the response
    Keyword arguments:
        user_input -- the user input from the input field
        keyword -- the user input type
    """
    # Instantiates a model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    # Creates a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly tour guide called Max."
                   "You are going to answer a series of questions from a visitor. "
                   "Answer the questions in english."),
        ("system", msg_dict[keyword]
         ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    # Creates a serpAPI tool
    search = SerpAPIWrapper()
    repl_tool = StructuredTool.from_function(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command."
                    "If you want to see the output of a value, you should print it out with `print(...)`."
                    "Input to this tool must be a SINGLE STRING",
        func=search.run,
    )
    # Creates a toolset
    tools = [repl_tool]
    # Instantiates an agent
    agent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools
    )
    # Invokes the agent
    response = agent_executor.invoke({
        "input": user_input
    })
    return response["output"]
