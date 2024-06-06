import pandas as pd
import numpy as np
import csv
import ast

from ftfy import fix_encoding
from nltk.tokenize import word_tokenize
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig
import torch.nn.functional as F
from transformers import pipeline
from scipy.special import softmax

import torch
print("Is GPU available-", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used -", device.type)

from emoji import demojize
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

absa = "yangheng/deberta-v3-base-absa-v1.1"
absa_tokenizer = AutoTokenizer.from_pretrained(absa, cache_dir="cache_huggingface_transformers/")
absa_model = AutoModelForSequenceClassification.from_pretrained(absa, cache_dir="cache_huggingface_transformers/")

absa_model = absa_model.to(device)

sent_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

sentiment_tokenizer = AutoTokenizer.from_pretrained(sent_model, cache_dir="cache_huggingface_transformers/")
sentiment_config = AutoConfig.from_pretrained(sent_model, cache_dir="cache_huggingface_transformers/")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sent_model, cache_dir="cache_huggingface_transformers/")

sentiment_model = sentiment_model.to(device)

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTP"
    elif token.startswith("#"):
        return token[1:]
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        elif token == "RT" or lowercased_token == "rt":
            return ""
        elif token == "#":
            return ""
        else:
            return token

def normalizeTweet(tweet):
    tokens = tweet_tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    return " ".join(normTweet.split())

def save_df_csv(df, path):
    
    cols = df.columns
    data = df.values.tolist()
    
    with open(path, mode='w', newline='',encoding="utf8") as file:
        
        writer = csv.writer(file)
        writer.writerow(cols)
        writer.writerows(data)  

def print_deberta_examples():

    sentence = "We had a great experience at the restaurant, food was delicious, but the service was kinda bad"
    print(f"Sentence: {sentence}")
    print()

    # ABSA of "food"
    aspect = "food"
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    inputs = inputs.to(device)
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.cpu().detach().numpy()[0]
    print(f"Sentiment of aspect '{aspect}' is:")
    for prob, label in zip(probs, ["negative", "neutral", "positive"]):
        print(f"Label {label}: {prob}")
    print()

    # ABSA of "service"
    aspect = "service"
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    inputs = inputs.to(device)
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.cpu().detach().numpy()[0]
    print(f"Sentiment of aspect '{aspect}' is:")
    for prob, label in zip(probs, ["negative", "neutral", "positive"]):
        print(f"Label {label}: {prob}")
    print()

def get_aspect_score(sentence, aspect):
    
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    inputs = inputs.to(device)
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.cpu().detach().numpy()[0]

    rest = {}
    
    for prob, label in zip(probs, ["negative", "neutral", "positive"]):
        rest[label] = np.round(float(prob), 4) #prob
    
    Category = None
    
    l = max(rest, key=rest.get)
    
    if l=='negative':
        Category = 1
    elif l=='neutral':
        Category = 2
    elif l=='positive':
        Category = 3
        
    return rest, Category
    
def get_aspect_overall(text):
    
    encoded_input = sentiment_tokenizer(text, return_tensors='pt', truncation = True, max_length = 512, padding = False)
    encoded_input = encoded_input.to(device)
    output = sentiment_model(**encoded_input)
    scores = output[0][0].cpu().detach().numpy()
    scores = softmax(scores)
    
    rest = {'negative':0, 'neutral':0, 'positive':0}
    
    Category = None
    
    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = sentiment_config.id2label[ranking[i]]
        s = scores[ranking[i]]
        rest[l] = np.round(float(s), 4)
        if i==0:
            if l=='negative':
                Category = 1
            elif l=='neutral':
                Category = 2
            elif l=='positive':
                Category = 3
        
    return rest, Category

def check_category (aspect, aspect_categories):
    
    for k in aspect_categories.keys():
        if aspect in aspect_categories[k]:
            return k

def read_file(read_path, text_column_name):
    df = pd.read_csv(read_path)
    print("Original File Shape -", df.shape)
    df = df.dropna(subset=[text_column_name]).reset_index(drop=True)
    print("Processing Text......")
    df["processed_text"] = df[text_column_name].apply(normalizeTweet)
    print("Processed File Shape -", df.shape)
    return df

def deberta_main(df_og, aspects_list, aspect_categories):

    length = df_og.shape[0]
    
    for i in range(length):
    
        example = df_og.loc[i,"processed_text"]
            
        text = example.lower()
        txt1 = word_tokenize(text)
        
        for a in aspects_list:
            if a in txt1:

                rest, cat = get_aspect_score(example, a)

                index = check_category (a, aspect_categories)

                if len(df_og.loc[i,'Aspect_Category_'+str(index)])==0:
                    df_og.loc[i,'Aspect_Category_'+str(index)] = a
                    df_og.loc[i,'Aspect_Category_'+str(index)+'_Sentiment (1-Negative, 2-Neutral, 3-Positive)'] = str(cat)
                    df_og.loc[i,'Aspect_Category_'+str(index)+'_Scores'] = str(rest)
                else:
                    df_og.loc[i,'Aspect_Category_'+str(index)] += ", " + a
                    df_og.loc[i,'Aspect_Category_'+str(index)+'_Sentiment (1-Negative, 2-Neutral, 3-Positive)'] += ", " + str(cat)
                    df_og.loc[i,'Aspect_Category_'+str(index)+'_Scores'] += ", " + str(rest)
                          
        rest, cat = get_aspect_overall(example)
        df_og.loc[i,"Overall Sentiment Score"] = str(rest)
        df_og.loc[i,"Overall Sentiment (1-Negative, 2-Neutral, 3-Positive)"] = str(cat)

        if i%100==0:
            print("Progress = "+str(i)+"/"+str(length))
    
    print(str(length) + " rows processed. Completed.")

def dict_to_list(d):
    
    l = [d['negative'], d['neutral'], d['positive']]
    
    return l

def fixing_deberta(df, last_column_number):
    for i in range(df.shape[0]):
        for j in range(last_column_number,df.shape[1]-2,3):
            try:
                st = df.iloc[i,j].split(", ")
            except:
                continue
            
            score, cat = combination_deberta(df.iloc[i,j+1])
            df.iloc[i,j+1] = score
            df.iloc[i,j+2] = cat
              
    return df

def combination_deberta(confidence):
    
    l = ast.literal_eval(confidence)

    if isinstance(l, dict):

        l_1 = dict_to_list(l)
        l_2 = np.asarray(l_1)
        
        return np.max(l_2), np.argmax(l_2)+1
    
    elif isinstance(l, tuple):
        
        l_sum = np.asarray(dict_to_list(l[0]))
        
        for x in range(1,len(l)):
            l_sum = l_sum + np.asarray(dict_to_list(l[x]))
        
        l_2 = np.asarray(l_sum)
        l_2 = l_2/len(l)
        
        return np.max(l_2), np.argmax(l_2)+1
    
def deberta_results(df_deberta, last_column_number, number_aspect_cats):

    df_deberta_2 = fixing_deberta(df_deberta)

    df_1 = df_deberta_2.iloc[:,last_column_number:last_column_number+3]
    df_1 = df_1.dropna()
    df_1["Aspect_Category_1"] = "Category_1"

    for i in range(1,number_aspect_cats-1):

        df_2 = df_deberta_2.iloc[:,last_column_number+i*3:last_column_number+3+i*3]
        
        new_cols = {x: y for x, y in zip(df_2.columns, df_1.columns)}
        df_2 = df_2.rename(columns=new_cols)
        
        df_2 = df_2.dropna()
        
        df_2["Aspect_Category_1"] = "Category_"+str(i+1)
        
        df_1 = pd.concat([df_1, df_2], ignore_index=True)

    print(df_1["Aspect_Category_1"].value_counts())
    print(df_1["Aspect_Category_1_Sentiment (1-Negative, 2-Neutral, 3-Positive)"].value_counts())

    df_3 = df_1.groupby(by=["Aspect_Category_1","Aspect_Category_1_Sentiment (1-Negative, 2-Neutral, 3-Positive)"]).agg(['count','mean'])

    return df_deberta_2, df_3

def main(read_path, text_column_name, aspect_categories, save_path_1, save_path_2, save_path_3):

    df_og = read_file(read_path, text_column_name)

    last_column_number = df_og.shape[1]
    number_aspect_cats = len(aspect_categories.keys())

    aspects_list = []
    for k in aspect_categories.keys():
        aspects_list+=aspect_categories[k]

    for x in range(number_aspect_cats):
        df_og['Aspect_Category_'+str(x+1)] = ""
        df_og['Aspect_Category_'+str(x+1)+'_Scores'] = ""
        df_og['Aspect_Category_'+str(x+1)+'_Sentiment (1-Negative, 2-Neutral, 3-Positive)'] = ""

    df_og["Overall Sentiment Score"] = ""
    df_og["Overall Sentiment (1-Negative, 2-Neutral, 3-Positive)"] = ""

    df_og_processed = deberta_main(df_og, aspects_list, aspect_categories)
    print("Sentiments & Confidence Scores Generated")

    save_df_csv(df_og_processed, save_path_1)
    print("File with sentiment & confidence scores saved. Shape=" + str(df_og_processed.shape))

    df_og_2, df_og_3 = deberta_results(df_og, last_column_number, number_aspect_cats)

    print("File with confidence scores combined saved. Shape=" + str(df_og_2.shape))
    save_df_csv(df_og_2, save_path_2)
    print("File with group counts of categories & sentiment. Shape=" + str(df_og_3.shape))
    save_df_csv(df_og_3, save_path_3)