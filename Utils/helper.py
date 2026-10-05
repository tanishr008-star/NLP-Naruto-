from glob import glob 
import pandas as pd
import os
def load_dataset():
    files = glob(r"C:\Users\shawa\Downloads\football-analysis\nlp-tv-analysis\data\Subtitles\*.ass")
    scripts = []
    episodes = []
    for path in files:
        with open(path,'r',encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [ ",".join(line.split(",")[9:]) for line in lines]
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)
        episode = int(path.split('-')[-1].split('.')[0].strip())
        scripts.append(script)
        episodes.append(episode)
    df = pd.DataFrame({"epsiode_num":episodes, "script":scripts})

    print(df)
    return df
