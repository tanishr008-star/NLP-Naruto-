from transformers import pipeline
import torch
import os
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import numpy as np
from utils import load_dataset
nltk.download('punkt')
nltk.download('punkt_tab')

def load_model():
    theme_classifier = pipeline(
        "zero-shot-classification",
        model = "facebook/bart-large-mnli", 
        device = 0 if torch.cuda.is_available() else 'cpu'
    )
    return theme_classifier


def get_themes_inference(script, theme_classifier, theme_list):
    script_sentences = sent_tokenize(script)
        # Batch Sentence
    sentence_batch_size=20
    script_batches = []
    for index in range(0,len(script_sentences),sentence_batch_size):
        sent = " ".join(script_sentences[index:index+sentence_batch_size])
        script_batches.append(sent)
        
        # Run Model
    theme_output = theme_classifier(
        script_batches[:2],
        theme_list,
        multi_label=True
        )

        # Wrangle Output 
    themes = {}
    for output in theme_output:
        for label,score in zip(output['labels'],output['scores']):
            if label not in themes:
                themes[label] = []
            themes[label].append(score)

    themes = {key: np.mean(np.array(value)) for key,value in themes.items()}

    return themes

def get_dataset_with_themes(theme_classifier, theme_list, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        return df
    

    df = load_dataset()
    df= df.head(2)
    print(df)
        # Run Inference
    output_themes = df['script'].apply(lambda x: get_themes_inference(x,theme_classifier, theme_list))

    themes_df = pd.DataFrame(output_themes.tolist())
    df[themes_df.columns] = themes_df

    print("df",df)

        # Save output
    if save_path is not None:
        df.to_csv(save_path,index=False)
        
    return df
