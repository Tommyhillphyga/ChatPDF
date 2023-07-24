import os 
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat your PDF")
    st.header("Chat your PDF")
    st.sidebar.title("Developed by Tomiwa Samuel")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    


    pdf = st.sidebar.file_uploader("upload your file", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        for pages in pdf_reader.pages:
            text = ""
            text+=pages.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 150,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        user_query = st.text_input('Ask your question')


        if user_query:
            docs = knowledge_base.similarity_search(user_query)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents = docs, question= user_query )
            st.write(result)




    
    





if __name__ == "__main__":
    main()