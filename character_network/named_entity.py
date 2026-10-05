import spacy
from nltk import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_dataset


def load_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

def get_inference(script):
    nlp_model = load_model()
    sentences = sent_tokenize(script)
    output = []
    for sentence in sentences:
        doc = nlp_model(sentence)
        ners = set()
        for entity in doc.ents:
            if entity.label_ =="PERSON":
                full_name = entity.text
                first_name = entity.text.split(" ")[0]
                first_name = first_name.strip()
                ners.add(first_name)
        output.append(ners)
    return output



def named_entity(save_path):
    if save_path is not None and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df['characters'] = df['characters'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
        return df
    
    df = load_dataset()
    df = df.head(10)

    df["characters"] = df['script'].apply(get_inference)

    if save_path is not None:
        df.to_csv(save_path,index=False)

    return df

    
