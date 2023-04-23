from pathlib import Path
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
import os

def make_dataset(path):
  dataset = [ ]

  for filepath in Path(path).glob("**/*.jpg"):
    breed_name = str(filepath).split("/")[-2]
    dataset.append([str(filepath), breed_name])

  dataset = np.array(dataset)
  return dataset

def split_dataset(dataset):
  # 0.25
  X_train, X_test, y_train, y_test = train_test_split(dataset[:,0], dataset[:,1], test_size=0.2, stratify=dataset[:,1])
  # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
  return X_train, X_test, y_train, y_test

def make_folders(DATA_PATH, y_test):
  if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

  if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
    os.makedirs(os.path.join(DATA_PATH, "train"))
    os.makedirs(os.path.join(DATA_PATH, "test"))
    # os.makedirs(os.path.join(DATA_PATH, "validation"))

    for breed_name in set(y_test):
      os.makedirs(os.path.join(DATA_PATH, "train", breed_name))
      os.makedirs(os.path.join(DATA_PATH, "test", breed_name))
      # os.makedirs(os.path.join(DATA_PATH, "validation", breed_name))

def move_dataset(X, y, mode="train", DATA_PATH="dataset"):
  for filepath, taregt_dir in zip(X.tolist(), y.tolist()):
    filename = filepath.split("/")[-1]
    source_path = filepath
    target_dir = os.path.join(DATA_PATH, mode, taregt_dir, filename)
    shutil.copy(source_path, target_dir)

def main(path="../Images/", DATA_PATH="dataset"):
  dataset = make_dataset(path)
  X_train, X_test, y_train, y_test = split_dataset(dataset)
  make_folders(DATA_PATH, y_test)
  move_dataset(X_train, y_train, mode="train", DATA_PATH=DATA_PATH)
  move_dataset(X_test, y_test, mode="test", DATA_PATH=DATA_PATH)
  # move_dataset(X_val, y_val, mode="validation", DATA_PATH=DATA_PATH)

if __name__ == '__main__':
  main(path="/home/chaewon/workspace/DL/Dog_breed/Images/", DATA_PATH="dataset")