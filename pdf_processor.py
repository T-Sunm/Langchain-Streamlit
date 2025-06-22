# pdf_processor.py
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
def extract_text_from_pdf(pdf_file):
  pdf_reader = PdfReader(pdf_file)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text

def process_text(text):
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

  # Chuyển đổi các chuỗi văn bản thành đối tượng Document
  documents = [Document(page_content=chunk) for chunk in chunks]
  embeddings = HuggingFaceEmbeddings()

  # Xây dựng cơ sở dữ liệu vector sử dụng FAISS để lưu trữ và truy vấn các embeddings
  knowledge_base = Chroma.from_documents(
      documents=documents, embedding=embeddings)

  retriever = knowledge_base.as_retriever()

  return retriever

def format_docs(docs):
  return "\n\n". join(doc . page_content for doc in docs)
