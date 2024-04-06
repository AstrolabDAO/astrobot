import hashlib
import os
from pathlib import Path
import pickle
import sys
from itertools import chain

def flatten(lsts):
    return list(chain(*lsts))

def itertools_chain_from_iterable(lsts):
    return list(chain.from_iterable(lsts))
class SuppressStdout:
  def __enter__(self):
    self._original_stdout = sys.stdout
    self._original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._original_stdout
    sys.stderr = self._original_stderr

def save(data, file: Path|str):
  file = Path(file)
  file.parent.mkdir(parents=True, exist_ok=True)
  if isinstance(data, str):
    data = data.encode()
  if not isinstance(data, bytes):
    data = pickle.dumps(data)
    file = file.with_suffix(".pkl")
  with open(file, "wb") as f:
    f.write(data)

def load(file: Path|str):
  file = Path(file)
  if not file.exists():
    file = file.with_suffix(".pkl")
  if file.exists():
    with open(file, "rb") as f:
      if file.suffix == ".pkl":
        return pickle.load(f)
      return f.read()
  return None

def get_file_digest(file_path: Path) -> str:
    # Compute the MD5 hash of a file's content.
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
      buf = f.read()
      hasher.update(buf)
    return hasher.hexdigest()

def get_digest(doc: str|list[str]) -> str:
  if isinstance(doc, list):
    doc = "".join(doc)
  hasher = hashlib.md5()
  hasher.update(doc.encode())
  return hasher.hexdigest()
