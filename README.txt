Download the gpt4all model:
https://huggingface.co/mrgaang/aira/blob/main/gpt4all-converted.bin

and put in the folder you prefer, but remember to indicate it in the "model_path" in the code.

STEPS:
1. Load the GPT4All model after downloading from above link
2. Use Langchain to retrieve your documents after load them in the folder "Samples"
3. Run "Semantic_retrieval_knowhow.py" to split the documents in small chunks digestible by embeddings, and use FAISS to create your vector database with the embeddings
5. Run "db_loading.py" to perform a similarity search (semantic search) on your vector database based on the question you want to pass to GPT4All: this will be used as a context for your question
6. Feed the question and the context to GPT4All with Langchain and wait for the answer
