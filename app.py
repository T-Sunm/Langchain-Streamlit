from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def main():
  load_dotenv()
  st.set_page_config(page_title="Ask your PDF")
  st.header("Ask your PDF")

# upload file
  pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    # split into chunks

    # cấu hình
    text_splitter = CharacterTextSplitter(
        #   tách lấy từng câu
        separator="\n",
        # 1 lần lấy 1000 ký tự
        chunk_size=1000,
        # thụt lui sau 200 từ rồi lấy
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)


if __name__ == "__main__":
  main()
