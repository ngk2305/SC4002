import pandas as pd
import re


def load_glove_vectors(glove_file):
    word_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_vectors[word] = vector
    return word_vectors

def get_word_embeddings(text):
    embeddings = []
    for word in text.split():
        if word in glove_vectors:
            embeddings.append(glove_vectors[word])
    return embeddings

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def run(filename,productname):
    df = pd.read_csv(filename)
    df[text_column] = df[text_column].apply(clean_str)
    # Apply the function to the 'text_column' and create a new column for embeddings
    # print(get_word_embeddings('laugh'))
    df['word_embeddings'] = df[text_column].apply(get_word_embeddings)
    df.rename(columns={'label-coarse': 'numerical_label'}, inplace=True)
    # Add a new column with numerical representations
    selected_columns = ['numerical_label', 'word_embeddings']
    selected_df = df[selected_columns].astype('str')
    selected_df = selected_df.loc[selected_df['word_embeddings'] != '[]']
    selected_df.to_csv(productname, index=False)


text_column='text'
label_column= 'label-coarse'

if __name__ == "__main__":
    # Provide the path to your downloaded GloVe file
    glove_file = 'glove.6B.50d.txt'  # Replace with the correct file path
    glove_vectors = load_glove_vectors(glove_file)
    run('train.csv','train_vec.csv')
    run('test.csv', 'test_vec.csv')
    run('dev.csv', 'dev_vec.csv')

