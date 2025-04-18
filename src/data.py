import pandas as pd 
import tarfile
import urllib.request
from pathlib import Path


def load_housing_data():
       tarball_path = Path("datasets/housing.tgz")
       if not tarball_path.is_file():
              Path("datasets").mkdir(parents=True, exist_ok=True)
              url = "https://github.com/ageron/data/raw/main/housing.tgz"
              urllib.request.urlretrieve(url,tarball_path)
              print("Downloading housing.tgz...")
              with tarfile.open(tarball_path) as housing_tarball:
                     housing_tarball.extractall(path="datasets")
                     print("Downloading")
       return pd.read_csv(Path("datasets/housing/housing.csv"))