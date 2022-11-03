# -*- coding: utf-8 -*-

import pandas as pd
from itertools import chain

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences



df = pd.read_csv("dev_set_2.tsv", sep='\t',encoding="utf-8")
essayColumn = df[["essay"]].values
idColumn = df[["essay_id"]].values
r1d1Column = df[["rater1_domain1"]].values
r2d1Column = df[["rater2_domain1"]].values
r1d2Column = df[["rater1_domain2"]].values
r2d2Column = df[["rater2_domain2"]].values
r1d3Column = df[["rater1_domain3"]].values
r2d3Column = df[["rater2_domain3"]].values

#print(essayColumn)
#print(idColumn)

essays = list(chain.from_iterable(essayColumn))
ids = list(chain.from_iterable(idColumn))
#print(essays[0])
#print(ids[0])

temp_essays = []



for essay in essays:
    sentences = split_into_sentences(essay)

    whole_essay = " ".join(sentences[0:len(sentences)//2+1]) * 2
    temp_essays.append(whole_essay)
    
print(len(idColumn),len(essays))
print(len(temp_essays))

with open("dev_set_2_modded.tsv","w+") as f:
    for i in range(0,360):
        print(temp_essays[i]+"\n")
        f.write(str(ids[i])+"\t"+"2"+"\t"+'"'+temp_essays[i]+'"'+"\n")
