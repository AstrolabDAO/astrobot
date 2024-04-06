from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, ConfigDict, HttpUrl

from vector_store import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import Runnable

class Repository(BaseModel):
    name: str
    url: HttpUrl
    subfolder: str
    exclude: Optional[List[str]]

class DocumentFolder(BaseModel):
    name: str
    path: str
    match: Optional[str]

class ModelConfig(BaseModel):
    name: str
    params: Optional[dict[str, Any]]

class DataLoadingConfig(BaseModel):
  repositories: Optional[List[Repository]]
  folders: Optional[List[DocumentFolder]]
  digests: Optional[dict[str, str]]
  digest: Optional[str]

class EmbeddingConfig(BaseModel):
    model: ModelConfig
    splitter: Optional[dict[str, Any]]
    data: DataLoadingConfig

class TrainingConfig(BaseModel):
    model: ModelConfig
    splitter: Optional[dict[str, Any]]
    data: DataLoadingConfig

class PromptConfig(BaseModel):
    template: str

class InferenceConfig(BaseModel):
    model: ModelConfig
    prompt: PromptConfig

class Config(BaseModel):
  training: Optional[TrainingConfig]
  tuning: Optional[EmbeddingConfig]
  inference: InferenceConfig

@dataclass
class RuntimeEnv():
    config: Config
    store: Chroma # vector store for retriever
    llm: Ollama
    prompt: PromptTemplate
    rag_setup: Runnable
    inference_chain: Runnable
