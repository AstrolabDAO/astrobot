# NB: replace Chroma by Qdrant for better performance

from abc import ABC
from typing import List, Optional, Any

import chromadb
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma as _Chroma

# from https://gist.github.com/jasonjmcghee/99d547269743afbb8a876b77c3b4e9a3
class Chroma(_Chroma, ABC):

  @classmethod
  def from_cached_documents(
    cls,
    persist_directory: str,
    documents: List[Document],
    embedding: Optional[Embeddings] = None,
    ids: Optional[List[str]] = None,
    collection_name: str = _Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
    client_settings: Optional[chromadb.config.Settings] = None,
    **kwargs: Any,
  ) -> _Chroma:
    client = chromadb.PersistentClient(path=persist_directory)
    collection_names = [c.name for c in client.list_collections()]

    if collection_name in collection_names:
      return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
        client_settings=client_settings,
      )

    return Chroma.from_documents(
      documents=documents,
      embedding=embedding,
      ids=ids,
      collection_name=collection_name,
      persist_directory=persist_directory,
      client_settings=client_settings,
      **kwargs
    )
