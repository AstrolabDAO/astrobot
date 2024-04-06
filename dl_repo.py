import subprocess
import shutil
from pathlib import Path
from model import Repository

def dl_repo(repo: Repository, dst: str|Path='./data/tuning/repositories'):
  subfolder = repo.subfolder
  dst = Path(dst) / repo.name

  print(f"Processing repo [{repo.name}]...")

  # Clone or update the repository
  if not dst.exists():
    subprocess.run(['git', 'clone', str(repo.url), str(dst)])
  else:
    # subprocess.run(['git', '-C', str(dst), 'pull'])
    return

  # Now, we adjust our path to only keep the subfolder contents if specified
  if repo.subfolder and repo.subfolder != "/":
    full_subfolder = dst / subfolder.strip("/")
    temp_path = dst.with_suffix('.tmp')

    # Move the subfolder to a temporary location
    if temp_path.exists():
      shutil.rmtree(temp_path)
    full_subfolder.rename(temp_path)

    # Remove the original repo contents
    shutil.rmtree(dst)

    # Move back from temporary location to the original repo path
    temp_path.rename(dst)

  for exclude in repo.exclude:
    exclude_path = dst / exclude
    if exclude_path.exists():
      shutil.rmtree(exclude_path)

  print(f"Completed processing repo [{repo.name}]")

def dl_repos(repos: list[Repository]):
  for repo in repos:
    dl_repo(repo)
