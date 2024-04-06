# Load YAML configuration
import yaml
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag import load_store
from dl_repo import dl_repos
from model import Config, RuntimeEnv

def start_model(env: RuntimeEnv):

  while True:
    query = input("\nUser> ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    env.inference_chain.invoke(query)

def get_config() -> Config:
  with open('./config.yml', 'r') as file:
    return Config(**yaml.safe_load(file))

def setup_runtime() -> RuntimeEnv:
  config = get_config()
  dl_repos(config.tuning.data.repositories)
  store = load_store(config)
  llm = Ollama(
    model=config.inference.model.name,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    **config.inference.model.params)

  prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=config.inference.prompt.template)

  rag_setup = RunnableParallel(
      {"context": store.as_retriever(), "question": RunnablePassthrough()}
  )
  inference_chain = rag_setup | prompt | llm | StrOutputParser()
  return RuntimeEnv(config, store, llm, prompt, rag_setup, inference_chain)

if __name__ == '__main__':
  env = setup_runtime()
  start_model(env)
