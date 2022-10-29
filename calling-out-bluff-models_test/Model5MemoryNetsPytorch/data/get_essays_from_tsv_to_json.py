#import pandas for getting the essays into an array
import pandas as pd
#import nltk for breaking the paragraph into sentences
import nltk.data
#import json to dump all of the json data into a json file
import json


df = pd.read_csv("C:\\Users\\viksp\Documents\\Folder_of_Folders\\Polygence_code\\calling-out-bluff-models_test\\Model5MemoryNetsPytorch\\data\\12_scored_essays.tsv", sep='\t',encoding="utf-8")
categoryColumn = df[["essay"]].values

print(categoryColumn)

# categoryList = []
# for line in categoryColumn:
#     categoryColumn.append(line)
# print(categoryList[0:10])

print()
print()


#split the paragraph into sentences then generate the appropriate json data, RETURNS AN ARRAY OF JSON DATA IN THE BELOW FORMAT
def gen_json(test_str): #test_str is the input string
    
    '''

    The format of the fixtures.json is 
    {"label": "BACKGROUND", "sentence": "sentence content .\n"}

    Notice the space between the content and the period/exclamation mark + new line character.

    '''
    #get the data
    data = test_str
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #load nltk stop words
    sentences = tokenizer.tokenize(data) #tokenize the data, aka spliting the data into sentences
    #print('\n-----\n'.join(sentences)) #print the data, not used in the final result
    #print(sentences)
    json_arr=[]
    
    #test_str = "This is the FBI." #test string, not used in the final result
    for sentence in sentences:
        #print(sentence) #print the data, not used in the final result 
        sentence = sentence[0:len(sentence)-2] + " "+ "." + "\n" #add a space between the last two characters, then add a newline character 
        #print(test_str) #print the string, not used in the final result

        jason_data = {"label" : "BACKGROUND" , "sentence" : sentence}
        json_arr.append(jason_data)

        #print(sentence) #print the string, not used in the final result
    #print(test_str) #print the string, not used in the final result
    
    return json_arr


for i in range(0,1):
    json_temp_arr = gen_json(categoryColumn[i][0])
    with open('data.json', 'a', encoding='utf-8') as f:
        for jsn in json_temp_arr:
            json.dump(jsn,f)
            f.write("\n")
    
#print('\n-----\n'.join(gen_json(categoryColumn[0][0])))