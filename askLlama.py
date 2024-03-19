import argparse
from os import getenv
from os.path import exists
from dotenv import load_dotenv
from time import sleep
from llama_index import download_loader
from llama_index import download_loader, GPTVectorStoreIndex, StorageContext, ServiceContext, load_index_from_storage

def load_or_create_db_index(directory):

  service_context = ServiceContext.from_defaults(chunk_size_limit=512)

  if(exists(directory + "/../llamadb")):
      print("Loading vector database from " + directory + "/../llamadb")
      return  load_index_from_storage(
        service_context=service_context,    
        storage_context=StorageContext.from_defaults(persist_dir=directory + "/../llamadb")
        )

  SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

  loader = SimpleDirectoryReader('./documentos', file_extractor={
    ".pdf": "UnstructuredReader",
    ".html": "UnstructuredReader",
    ".eml": "UnstructuredReader",
    ".pptx": "PptxReader"
  })

  documents = loader.load_data()
  vectordb_index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
  vectordb_index.storage_context.persist(persist_dir="./llamadb")
  return vectordb_index
  

def process_dir(dir):
  
  vectordb_index = load_or_create_db_index(dir)

  query_engine = vectordb_index.as_query_engine(similarity_top_k=8)

  while(True):
      print("Ask Llama:\n")
      user_query = input()
      print(query_engine.query(user_query))
      sleep(0.2)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--file", help="File to process")
  parser.add_argument("--dir", help="directory with files to process")

  load_dotenv()

  if((parser.parse_args().dir is None)):
    print("Please, specify at least of the option: --file ")
    exit()

  if(getenv("OPENAI_API_KEY") is None or getenv("OPENAI_API_KEY") == ""):
    print("Please, set your OPENAI_API_KEY in a .env file")
    exit()

  print("OpenAI API Key set.")
  
  if(parser.parse_args().dir):
    process_dir(parser.parse_args().dir)

      