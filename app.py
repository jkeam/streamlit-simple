from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st

def document_data(query, chat_history):
   loader = TextLoader("data.txt")
   documents = loader.load()
   text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
   docs = text_splitter.split_documents(documents)
   embeddings = OpenAIEmbeddings()
   db = FAISS.from_documents(docs, embeddings)

   qa = ConversationalRetrievalChain.from_llm(
      llm=OpenAI(),
      retriever= db.as_retriever()
   )

   return qa({"question":query, "chat_history":chat_history})

if __name__ == '__main__':
   st.header("JonBot")
   prompt = st.chat_input("Enter your questions here")

   if "user_prompt_history" not in st.session_state:
     st.session_state["user_prompt_history"]=[]
   if "chat_answers_history" not in st.session_state:
     st.session_state["chat_answers_history"]=[]
   if "chat_history" not in st.session_state:
     st.session_state["chat_history"]=[]
   if "openai_model" not in st.session_state:
     st.session_state["openai_model"] = "gpt-3.5-turbo"

   if prompt:
      with st.spinner("Generating......"):
         output=document_data(query=prompt, chat_history=st.session_state["chat_history"])
         st.session_state["chat_answers_history"].append(output['answer'])
         st.session_state["user_prompt_history"].append(prompt)
         st.session_state["chat_history"].append((prompt,output['answer']))

   if st.session_state["chat_answers_history"]:
      for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
         message1 = st.chat_message("user")
         message1.write(j)
         message2 = st.chat_message("assistant")
         message2.write(i)
