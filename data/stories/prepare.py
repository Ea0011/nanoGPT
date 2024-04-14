import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# good number to use is ~order number of cpu cores // 2
num_proc = 8

### Load tiktoken gpt2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2") # This uses the tiktoken lib and gets gpt2 tokenizer

### Write a function to tokenize the text
def process(example):
    ### Get the text column
    text = example["text"]

    ### YOUR CODE GOES HERE

    ### Add <|endoftext|> token to the end of each text: FYI the id is 50256, if you wish to verify if the token is properly added to the ids after tokenization

    ### Tokenize by calling tiktoken encode function. Pass in allowed_special=True to tokenize the "<|endoftext|>" token as well

    ### Construct a result dict with two keys: ids and len. Ids are token ids and len is the length of the token ids

    ### Return the result
    return out

if __name__ == '__main__':
    ### Load the "deven367/babylm-100M-children-stories" dataset from huggingface

    ### YOUR CODE GOES HERE
    dataset = ...
    print(dataset)

    ### tokenize the dataset by using .map function

    ### YOUR CODE GOES HERE
    tokenized = ...
    
    ### Take the ids of the tokenized datasets. You can take only a small sample if you wish

    ### YOUR CODE GOES HERE
    train_ids = ...
    val_ids = ...

    ### Concatenate all ids into 1 large list. Do this both for train and val ids.

    ### YOUR CODE GOES HERE
    train_ids = ...
    val_ids = ...

    ### Export the prepared and tokenized datasets as binary files. These will be used for training.
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))