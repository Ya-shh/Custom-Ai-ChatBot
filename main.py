
import logging
import sys
from sentence_transformers import SentenceTransformer
import torch 
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index import  ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("path/to_your_directory").load_data()


llm = LlamaCPP(
        model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf',
        model_path=None, 
        temperature=0.7,
        max_new_tokens=512,
        context_window=4096,
        generate_kwargs={'top_p': 1, 'top_k': 50},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )



embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)




service_context = ServiceContext.from_defaults(
    chunk_size=16000,
    llm=llm,
    embed_model=embed_model,

)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    embedding_dim=768,
    index_type='HNSW',
    hnsw_ef_construction=200,
    hnsw_m=16,
    batch_size=64,
)

query_engine = index.as_query_engine()

while True:
    query = input("Please enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break

    try:

        response = query_engine.query(query)


        if response:
            print("Response received:")


            print(response)
        else:
            print("No relevant documents found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
      

