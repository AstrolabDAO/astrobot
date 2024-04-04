from dataclasses import dataclass
from pathlib import Path

@dataclass
class Repository:
  name: str
  url: str
  subfolder: str
  exclude: list[str]

@dataclass
class DocumentFolder:
  name: str
  path: Path
  match: str

@dataclass
class Config:
  embedding_model: str
  inference_model: str
  prompt_template: str
  repositories: list[Repository]
  folders: list[DocumentFolder]
