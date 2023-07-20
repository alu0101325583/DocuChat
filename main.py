
import re
import argparse
from os import system, name
from time import sleep
from os import getenv, walk
from os.path import isfile, join, exists
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader

from langchain.prompts.prompt import PromptTemplate

from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


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
    dbPath = directory + "/vectordb"
    if exists(dbPath):
        print("Loading vector database from " + dbPath)
        return Chroma(persist_directory=dbPath, embedding_function=embeddings)

    loaders = get_loaders(directory)

    print("Creating vector database in " + dbPath)

    fil = {"persist_directory": dbPath,
           "embedding_function": embeddings}  # ,"chroma_db_impl": "duckdb+parquet"}
    
    index_creator = VectorstoreIndexCreator(vectorstore_kwargs=fil)
    vector_index = index_creator.from_loaders(loaders)
    return vector_index.vectorstore


def process_dir(directory="./documents"):

    _template = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente. Ten en cuenta que si el historial esta vacio, la pregunta reformulada sera la pregunta inicial.

Historial de chat:
{chat_history}
Pregunta de seguimiento: {question}
Pregunta reformulada:"""

    vectordb = load_or_create_vectordb(directory)
    llm = ChatOpenAI(temperature=0.1)

    document_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 8}),
        # memory=memory,
        # qa_prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        return_source_documents=True,
        condense_question_prompt=PromptTemplate.from_template(_template)
    )

    chat_history = []
    system("cls" if name == "nt" else "clear")
    print("Hola, preguntame algo sobre tus documentos:\n")

    while True:

        user_query = input()
        if user_query == "exit()":
            exit()

        result = document_qa({"question": user_query, "chat_history": chat_history})
        chat_history.append((user_query, result["answer"]))

        print("\nRespuesta: " + result["answer"])
        print("\n\nFuentes:")
        for source in result["source_documents"]:
            print(source.metadata)
        print("\nPreguntame: ")
        sleep(0.2)


if __name__ == '__main__':

    load_dotenv()
    if getenv("OPENAI_API_KEY") is None or getenv("OPENAI_API_KEY") == "":
        print("Please, set your OPENAI_API_KEY in a .env file")
        exit()
    print("OpenAI API Key set.")

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-d', '--directory', help='Directory with documents to process', required=False)
    args = parser.parse_args()

    if(args.directory is not None):
        process_dir(args.directory)
    else:
        process_dir()