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

# model_path = 'C:/Users/Ivan/AppData/Local/nomic.ai/GPT4All'
model_path = 'C:/Users/Ivan/AppData/Local/nomic.ai/GPT4All/gpt4all-converted.bin'
language = "en"

# model = gpt4all.GPT4All("gpt4all-converted.bin", model_path, model_type='llama', allow_download=True)
# model.generate("Once upon a time, ", streaming = True)
# Callback manager for handling the calls with the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Create the embedding object
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
# Create the GPT4ALL llm object
llm = GPT4All(model = model_path, callback_manager=callback_manager, verbose = True)

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

# Load our local index vector db
index = FAISS.load_local("my_faiss_index", embeddings)

# Create the prompt template
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Here my answer: """

############# INPUT ############################################################
question = "How to build an innovation process?" # Hardcoded question
# question = input("Enter your question: ") # User Input Question
################################################################################

################# HARD CODED QUESTION WITHOUT GPT ##############################
# docs = index.similarity_search(question)
# Get the matches best 3 results - defined in the function k=3
# print(f"The question is: {question}")
# print("Here the result of the semantic search on the index, without GPT4All...")
# print(docs[0])
################################################################################

# Creating the context
matched_docs, sources = similarity_search(question, index)
context = "\n".join([doc.page_content for doc in matched_docs])
# instantiating the prompt template and the GPT4ALL chain
prompt = PromptTemplate(template = template, input_variables = ["context", "question"]).partial(context = context)
llm_chain = LLMChain(prompt = prompt, llm = llm)
# Print the result
LLM_reply = llm_chain.run(question)
print(LLM_reply)

# Write full context  and answer in a file
with open("LLM_reply.txt", "w", encoding = "utf-8") as txt:
    txt.write(LLM_reply)
    txt.close()

content = []
linenum = 0
substr = "Here my answer:".lower()

with open('LLM_reply.txt', 'rt') as myfile:
    lines = myfile.readlines()
    for line in lines:
        if (line.lower().find(substr) != -1) or (linenum != 0):
            linenum += 1
            content.append(line)

LLM_reply_content = ' '.join(content)
print(LLM_reply_content)

file = open("LLM_reply_content.txt","w")
file.write(LLM_reply_content)
file.close()