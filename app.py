from dotenv import load_dotenv
import os
import streamlit as st
from pdf_processor import extract_text_from_pdf, process_text, format_docs
from langchain.chains.question_answering import load_qa_chain
from model import get_huggingface_llm
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
def main():

  llm = ChatOllama(model="mistral")
  load_dotenv()
  st.set_page_config(page_title="Ask your PDF")
  st.header("Ask your PDF")

# upload file
  pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
  if pdf is not None:
    text = extract_text_from_pdf(pdf_file=pdf)
    retriever = process_text(text)

    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = ({"context": retriever | format_docs,
                  "question": RunnablePassthrough()}
                 | prompt
                 | llm
                 | StrOutputParser()
                 )

  user_question = st.text_input("Ask a question about your PDF")
  if user_question:
    output = rag_chain.invoke(user_question)
    st.write(output)


if __name__ == "__main__":
  main()
