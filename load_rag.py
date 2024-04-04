# TODO: add support for web search/scraping
# https://python.langchain.com/docs/use_cases/web_scraping
# https://github.com/langchain-ai/web-explorer/blob/main/web_explorer.py

import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFium2Loader, UnstructuredEPubLoader, UnstructuredMarkdownLoader,
    UnstructuredRSTLoader, TextLoader, WebBaseLoader, MergedDataLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import SuppressStdout
from model import Config

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    # length_function=len,
    # is_separator_regex=False,
    separators=[
      "\n\n",
      "\n",
      ".",
      ",",
      " ",
      "\u200B",  # Zero-width space
      "\uff0c",  # Fullwidth comma
      "\u3001",  # Ideographic comma
      "\uff0e",  # Fullwidth full stop
      "\u3002",  # Ideographic full stop
      "",
    ],
)

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

# recursively crawl through folders, parse files, but do not index immediately
def get_docs(folder: str|Path):
  docs = []
  for root, _, files in os.walk(folder):
    for file_name in files:
      src = Path(root) / file_name
      loader = get_loader(src)
      if loader is not None:
        docs.append(loader(str(src)).load())
  return docs

def parse_docs(folders: list[str|Path]):
  loaders = []
  for folder in folders:
    loaders += get_docs(folder) # merge
  try:
    docs = MergedDataLoader(loaders=loaders).load()
    splits = splitter.split(docs)
  except Exception as e:
    print(e)
  return splits

def load_rag(config: Config) -> Chroma:
  folders=[f"./data/tuning/{folder.name}" for folder in config.folders + config.repositories]
  all_documents = parse_docs(folders)
  with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_documents, embedding=OllamaEmbeddings(model="gemma:2b"))
  return vectorstore
