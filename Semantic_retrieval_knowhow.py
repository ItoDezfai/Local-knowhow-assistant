from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader # Function for loading only TXT files
from langchain.text_splitter import RecursiveCharacterTextSplitter # Text splitter for create chunks
from langchain.document_loaders import UnstructuredPDFLoader # to be able to load the pdf files
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator # Vector Store Index to create our database about our knowledge
from langchain.embeddings import HuggingFaceEmbeddings # LLamaCpp embeddings from the Alpaca model
from langchain.vectorstores.faiss import FAISS # FAISS library for similarity search
import os # For interaction with the files
import datetime
import gpt4all

model_path = 'C:/Users/Ivan/AppData/Local/nomic.ai/GPT4All'
language = "en"

model = gpt4all.GPT4All("gpt4all-converted.bin", model_path, model_type='llama', allow_download=True)
# model.generate("Once upon a time, ", streaming = True) # Just to test if the model works

# Callback manager for handling the calls with the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Create the embedding object
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size = 256, chunk_overlap = 32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas = metadatas)

    return search_index

def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k = 3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

# Get the list of pdf files from the docs directory into a list format
pdf_folder_path = "./Samples"
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
num_of_docs = len(doc_list)
# Create a loader for the PDFs from the path
general_start = datetime.datetime.now()
print("Starting the loop...")
loop_start = datetime.datetime.now()
print("generating first vector database and then iterate with .merge_from")
loader = PyMuPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
# Load the documents with Langchain
docs = loader.load()
# Split in chunks
chunks = split_chunks(docs)
# Create the db vector index
db0 = create_index(chunks)
print("Main Vector database created. Start iteration and merging...")
for i in range(1, num_of_docs):
    print(doc_list[i])
    print(f"loop position {i}")
    loader =PyMuPDFLoader(os.path.join(pdf_folder_path,doc_list[i]))
    start = datetime.datetime.now()
    docs = loader.load()
    chucks = split_chunks(docs)
    dbi = create_index(chunks)
    print("start merging with db0...")
    db0.merge_from(dbi)
    end = datetime.datetime.now()
    elapsed = end - start
    # Total time
    print(f"completed in {elapsed}")
    print("------------------------------------")
loop_end = datetime.datetime.now()
loop_elapsed = loop_end - loop_start
print(f"All documents processed in {loop_elapsed}")
print(f"the database is done with {num_of_docs} subset of db index")
print("----------------------------------------------")
print(f"Merging completed")
print("----------------------------------------------")
print("Saving Merged Database Locally")
# Save the database locally
db0.save_local("my_faiss_index")
print("----------------------------------------------")
print(f"merged database saved as my_faiss_index")
general_end = datetime.datetime.now()
general_elapsed = general_end - general_start
print(f"All indexing completed in {general_elapsed}")
print("----------------------------------------------")