import glob
import nltk
nltk.download('omw-1.4')


import json
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from canary.argument_pipeline import download_model, analyse_file
from canary.corpora import load_corpus
from pathlib import Path

if __name__ == "__main__":
    
    # Download all models
    download_model("all")
    
    # Load version 1 of the essay corpus. 
    essays = load_corpus("argument_annotated_essays_1", download_if_missing=True)
    essays[0] += "\\"
    #print(essays)
    #print(Path(essays[0]))
    x=essays[0]
    while True:
        essays = glob.glob(x+"/*.txt")
        #print(essays)
        #a=essays[:]
        # Analyse the first essay
        # essays[0] contains the absolute path to the first essay text file
        analysis = analyse_file(essays[0])
        with open('result.json',"a") as fp:
            json.dump(analysis, fp)