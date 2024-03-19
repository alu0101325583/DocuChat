# streamlit run askyourpdf.py --server.maxUploadSize 200
# deepdocdetection para pdf loading (donut, tesseract)
# unstructured.io

import argparse
import re
from time import sleep
from os import listdir, getenv, walk, environ
from os.path import isfile, join, exists
from dotenv import load_dotenv

import PyPDF2 as PdfReader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader

from langchain.prompts.prompt import PromptTemplate


from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


def get_loaders(directory):
  loaders = []
  pattern = r'(?i).*\.(png|jpg|txt|html|pdf|eml|msg|rtf|epub|docx?|odt|pptx?)$'
  print("Documents found:")
  for root, dirs, files in walk(directory):
    for file in files:
      if (isfile(join(root, file))) & (re.match(pattern, file.lower()) is not None):
        loaders.append(UnstructuredFileLoader(join(root, file), mode="elements"))
        print(file)
  return loaders


def load_or_create_vectordb(directory):

  embeddings = OpenAIEmbeddings()

  if(exists(directory + "/../vectordb")):
    print("Loading vector database from " + directory + "/../vectordb")
    return Chroma(persist_directory=directory + "/../vectordb", embedding_function=embeddings)
  
  persist_directory = directory + "/../vectordb"
  print("Creating vector database in " + persist_directory)
  
  loaders = get_loaders(directory)

  fil = {"persist_directory": persist_directory, "embedding_function": embeddings } #,"chroma_db_impl": "duckdb+parquet"}
  index_creator = VectorstoreIndexCreator(vectorstore_kwargs=fil)
  vector_index = index_creator.from_loaders(loaders)
  return vector_index.vectorstore


def process_dir(directory = "/home/ercos/Escritorio/ITER_LLMs/documentos"):

  '''
  memory = ConversationEntityMemory(
    llm=llm, 
    #memory_key="chat_history", 
    return_messages=True
  )


  memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
  )
  '''

  _template = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente.
  Ten en cuenta que si el historial esta vacio, la pregunta reformulada sera la
   pregunta inicial.

Historial de chat:
{chat_history}
Pregunta de seguimiento: {question}
Pregunta reformulada:"""

  vectordb = load_or_create_vectordb(directory)
  llm = ChatOpenAI(temperature = 0.1)

  document_qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever= vectordb.as_retriever(search_kwargs={'k': 8}), 
    #memory=memory, 
    #qa_prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    return_source_documents=True,
    condense_question_prompt= PromptTemplate.from_template(_template)
  )

  chat_history = []

  print("Hola, preguntame algo sobre tus documentos:\n")

  while(True):

    user_query = input()

    result = document_qa({"question": user_query, "chat_history": chat_history})
    chat_history.append((user_query, result["answer"]))

    print("\nRespuesta: " + result["answer"])
    print("\n\nFuentes:")
    for source in result["source_documents"]:
      print(source.metadata)
    
    sleep(0.2)

def process_file(filename):

  # cargamos el pdf
  pdf_reader = PdfReader(filename)

  if(pdf_reader is None):
    print("Error al cargar el fichero")
    exit()

  pages = []
  for page in pdf_reader.pages:
    pages.append(page.extract_text())

  embeddings = OpenAIEmbeddings()
  vectordb = Chroma.from_documents(pages, embedding=embeddings) #, persist_directory=".")
  
  #vectordb.persist()
  llm = ChatOpenAI(temperature = 0)

  '''
  memory = ConversationEntityMemory(
    llm=llm, 
    memory_key="chat_history", 
    return_messages=True
  )
  '''

  memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
  )

  pdf_qa = ConversationalRetrievalChain.from_llm(
    llm, 
    vectordb.as_retriever(), 
    memory=memory, 
    #qa_prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE
  )

  while(True):

    print("\n\nIntroduce tu pregunta:\n")
    user_query = input()

    result = pdf_qa({"question": user_query})
    
    print("\nRespuesta: " + result["answer"])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="File to process")
    parser.add_argument("--dir", help="directory with files to process")

    load_dotenv()
    if((parser.parse_args().file is None) & (parser.parse_args().dir is None)):
      print("Please, specify at least one of the options: --file or --dir")
      exit()

    if((parser.parse_args().file is not None) & (parser.parse_args().dir is not None)):
      print("Please, specify only one of the options: --file or --dir")
      exit()

    if(getenv("OPENAI_API_KEY") is None or getenv("OPENAI_API_KEY") == ""):
      print("Please, set your OPENAI_API_KEY in a .env file")
      exit()
    
    print("OpenAI API Key set.")

    if(parser.parse_args().file):
      process_file(parser.parse_args().file)

    if(parser.parse_args().dir):
     process_dir(parser.parse_args().dir)

    