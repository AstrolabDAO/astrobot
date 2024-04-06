# TODO: add support for web search/scraping
# https://python.langchain.com/docs/use_cases/web_scraping
# https://github.com/langchain-ai/web-explorer/blob/main/web_explorer.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFium2Loader, UnstructuredEPubLoader, UnstructuredMarkdownLoader,
    UnstructuredRSTLoader, TextLoader, WebBaseLoader, MergedDataLoader
)
from vector_store import Chroma
from langchain_community.embeddings import OllamaEmbeddings, FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import SuppressStdout, flatten, get_digest, get_file_digest, load, save
from model import Config

loaders = {
  # '.html': WebBaseLoader,
  '.pdf': PyPDFium2Loader,
  '.epub': UnstructuredEPubLoader,
  '.md': UnstructuredMarkdownLoader,
  '.rst': UnstructuredRSTLoader,
  '.wrm': UnstructuredRSTLoader,
  '.txt': TextLoader,
  '.csv': TextLoader,
  '.sol': TextLoader,
  '.js': TextLoader,
  '.ts': TextLoader,
  '.py': TextLoader,
  '.rs': TextLoader,
  '.go': TextLoader,
  '.sql': TextLoader
}

# identify and create loader based on file extension
def get_loader(src: str|Path):
  return loaders.get(Path(src).suffix.lower())

def process_folder(folder_path, config: Config):
  digests = []
  docs = []
  def load_file(file_name):
    src = Path(folder_path) / file_name
    if src.is_file():
      file_hash = get_file_digest(src)
      digests.append(file_hash)
      config.tuning.data.digests[file_name] = file_hash
      cache = Path(f"./data/cache/{file_hash}-doc")
      cached_data = load(cache)
      if cached_data is not None:
        print(f"Loaded cached [{file_name}]<-{cache}")
        docs.append(cached_data)
      else:
        loader = loaders.get(src.suffix)
        if loader is not None:
          print(f"Loading [{file_name}] using {loader.__name__}...")
          data = loader(str(src)).load()
          print(f"Caching [{file_name}]->{cache}")
          save(data, cache)
          docs.append(data)
  # FIXME: multi-threading breaks the loaders (sync language detection?), max_workers=1 is a tmp fix
  with ThreadPoolExecutor(max_workers=1) as executor:
    tasks = {executor.submit(load_file, file_name): file_name for file_name in os.listdir(folder_path)}
  list(map(lambda future: future.result(), as_completed(tasks)))
  config.tuning.data.digest = get_digest(digests)
  return docs

def get_docs(folder: str|Path, config: Config):
  with ThreadPoolExecutor() as executor:
    tasks = {executor.submit(process_folder, Path(root), config): root for root, dirs, files in os.walk(folder) if files}
  docs = list(map(lambda future: future.result(), as_completed(tasks)))
  return flatten(docs)

def parse_docs(config: Config):

  folders=[f"./data/tuning/{folder.name}" for folder in config.tuning.data.folders + config.tuning.data.repositories]
  splitter = RecursiveCharacterTextSplitter(**config.tuning.splitter)
  docs = []

  with ThreadPoolExecutor() as executor:
    future_to_folder = {executor.submit(get_docs, Path(root), config): root for root in folders}

  for future in as_completed(future_to_folder):
    docs.extend(future.result())

  split_cache = Path(f"./data/cache/{config.tuning.data.digest}-split-docs")
  splits = load(split_cache)
  if splits is not None:
    print(f"Loaded {len(splits)} cached splits from {split_cache}")
  else:
    print(f"Splitting {len(docs)} docs ({splitter._chunk_size} tokens chunk size)...")
    # docs = MergedDataLoader(loaders=loaders).load() # load separately and merge for easier debugging
    splits = splitter.split_documents(flatten(docs))
    # except Exception as e:
    #   print(e)
    print(f"Caching {len(splits)} splits to {split_cache}...")
    save(splits, split_cache)
  return splits

def load_store(config: Config) -> Chroma:
  docs = parse_docs(config)
  vectorstore_cache = Path(f"./data/cache/{config.tuning.data.digest}-rag-store")
  vectorstore = load(vectorstore_cache)
  if vectorstore is not None:
    print(f"Loaded cached RAG vectorstore from {vectorstore_cache}")
  else:
    # with SuppressStdout():
    print(f"Loading RAG vectorstore from {len(docs)} splits...")
    vectorstore = Chroma.from_cached_documents(
      documents=docs,
      collection_name="rag-store",
      persist_directory=f"./data/cache/chroma-{config.tuning.model.name}-{config.tuning.data.digest}",
      embedding=FastEmbedEmbeddings( # OllamaEmbeddings(model=config.tuning.model.name)
        model_name=config.tuning.model.name,
        cache_dir=f"./data/cache/embeddings-{config.tuning.data.digest}", # model agnostic cache
        **config.tuning.model.params
      )
    )
  return vectorstore
