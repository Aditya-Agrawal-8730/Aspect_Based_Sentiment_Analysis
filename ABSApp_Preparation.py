import pandas as pd

from emoji import demojize
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer() 

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

def file_no_preprocessing(read_path, text_column_name, save_path):

    df = pd.read_csv(read_path)

    print("Original File Shape -", df.shape)
    print("Null " +text_column_name+  " rows removed -", df[df[text_column_name].isna()].shape[0])

    df_2 = df.copy()
    df_2 = df_2.dropna(subset=[text_column_name])
    df_2 = df_2[[text_column_name]]

    print("Processed File Shape -", df_2.shape)

    df_2.to_csv(save_path,header=None, index=False)

def file_preprocessing(read_path, text_column_name, save_path):

    df = pd.read_csv(read_path)

    print("Original File Shape -", df.shape)
    print("Null " +text_column_name+  " rows removed -", df[df[text_column_name].isna()].shape[0])

    df_2 = df.copy()
    df_2 = df_2.dropna(subset=[text_column_name])
    df_2["Processed_Text"] = df_2[text_column_name].apply(normalizeTweet)
    df_2 = df_2[["Processed_Text"]]

    print("Processed File Shape -", df_2.shape)

    df_2.to_csv(save_path,header=None, index=False)