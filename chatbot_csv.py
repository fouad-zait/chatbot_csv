import streamlit as st
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from openai import OpenAI
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import pandas as pd

#PUT THE PATH OF YOUR CSV HERE and YOUR CSV in data_csv
csv_path = "data_csv/Jeux-de-données.csv"

user_api_key = st.secrets['OPENAI_API_KEY']

general_system_template = r""" 
 ----You are a virtual assistant designed to answer all questions about this CSV file.
{context}
----
"""
general_user_template =  """
        ### context ###

        {context}

        ### instructions ### 
 Respond concisely using a list if necessary, 
reply in english.
  If the question is not about the CSV file, politely inform them that you are set 
  up to answer only questions about the given CSV . 
 Provide a conversational response. Answer the question without giving unnecessary details.
        ### question ###

         {question}

        ### response : ### 
        """
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
template = ChatPromptTemplate.from_messages(messages)

# Chargement des données CSV
loader = CSVLoader(file_path=csv_path, encoding="utf-8")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
faiss = FAISS.from_documents(data, embeddings)
faiss_retriever = faiss.as_retriever()

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
    retriever=faiss_retriever, combine_docs_chain_kwargs={'prompt': template}
)

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about the CSV data 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# streamlit run tuto_chatbot_csv.py
