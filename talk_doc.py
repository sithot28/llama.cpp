import sys
import os

from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, GPTVectorStoreIndex, PromptHelper, LLMPredictor, S
erviceContext
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


model_name="NousResearch/Nous-Hermes-Llama2-13b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
model = model.eval()
print("Model loaded......")

class ChatGLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, history = model.chat(tokenizer, prompt, history=[])
        # only return newly generated tokens
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "chatglm-6b"}

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

directory_path = "./docs"

max_input_size = 4096
num_outputs = 2000
max_chunk_overlap = 20
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

documents = SimpleDirectoryReader(directory_path).load_data()
print("document loaded.....")


embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=ChatGLM())
print("setup llm_predictor.....")


service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)
index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
index.storage_context.persist()

storage_context = StorageContext.from_defaults(persist_dir='./storage')
index = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("<Who is president director of BSI>?")
print(response.response)
