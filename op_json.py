from gensim.models import Word2Vec;
import os;
import json;

def readAllJsonFile(dirpath):
    dataset = []
    for filename in os.listdir(dirpath):
        if filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset.append(data)
    return dataset
