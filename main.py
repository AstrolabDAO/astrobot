# Load YAML configuration
import yaml
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from load_rag import load_rag
from dl_repo import dl_repos
from model import Config, Repository, DocumentFolder

def start_model(config: Config):

  vectorstore = load_rag(config)
  llm = Ollama(model=config.inference_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

  QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=config.prompt_template,
  )

  while True:
    query = input("\nUser> ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=vectorstore.as_retriever(),
      chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain({"query": query})

if __name__ == '__main__':
  with open('./data/tuning/index.yml', 'r') as file:
    raw = yaml.safe_load(file)
    repositories = list(map(lambda r: Repository(**r), raw['repositories']))
    folders = list(map(lambda f: DocumentFolder(**f), raw['folders']))
    del raw['repositories']
    del raw['folders']
    config = Config(repositories=repositories, folders=folders, **raw)
  dl_repos(config.repositories)
  start_model(config)
