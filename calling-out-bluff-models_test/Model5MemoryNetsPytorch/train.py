import datalib as data
from model import MANM
from metric import kappa
import argparse
import time
import numpy as np
from torch import optim
import torch

import re
import pandas as pd
import sys

import json as json

import logging
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, SCORERS

import numpy as np
from scipy import stats


def load_data(path: str, set_id: int):
    """
    Loads lists of essays, scores and IDs
    """
    all_data = pd.read_csv(open(f"./data/{path}.tsv",errors="replace"), sep="\t", header=0)
    contents = all_data[all_data["essay_set"] == set_id]["essay"]
    essay_ids = all_data[all_data["essay_set"] == set_id]["essay_id"].values
    if not path == "test":
        essay_scores = all_data[all_data["essay_set"] == set_id]["domain1_score"].values
    else:
        essay_scores = [-1] * len(essay_ids)
    essay_contents = contents.apply(lambda x: tokenize(clean_str(x))).tolist()
    return essay_contents, list(essay_scores), list(essay_ids)

def load_test_data(path: str, set_id: int):
    """
    Loads lists of essays, scores and IDs
    """
    all_data = pd.read_csv(open(f"./data/{path}.tsv",errors="replace"), sep="\t", header=0)
    contents = all_data[all_data["essay_set"] == set_id]["essay"]
    essay_ids = all_data[all_data["essay_set"] == set_id]["essay_id"].values
    if not path == "test":
        essay_scores = all_data[all_data["essay_set"] == set_id]["domain1_score"].values
    else:
        essay_scores = [-1] * len(essay_ids)
    essay_contents = contents.apply(lambda x: tokenize(clean_str(x))).tolist()
    return essay_contents, list(essay_ids)


def all_vocab(train, dev, test):
    """
    Returns the vocabulary
    """
    data = train + dev + test
    words = []
    for item in data:
        words.extend(item)
    return set(words)


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    >> tokenize('I don't know')
        ['I', 'don', '\'', 'know']
    """
    return [x.strip() for x in re.split("(\W+)", sent) if x.strip()]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_glove(w_vocab, token_num=6, dim=50):
    word2vec = []
    word_to_index = {}
    # first word is nil
    word2vec.append([0] * dim)
    count = 1
    with open(
        "./glove/glove." + str(token_num) + "B." + str(dim) + "d.txt", encoding="utf-8"
    ) as f:
        for line in f:
            l = line.split()
            word = l[0]
            if word in w_vocab:
                vector = list(map(float, l[1:]))
                word_to_index[word] = count
                word2vec.append(vector)
                count += 1
    print("==> glove is loaded")
    print(f"word2vec total size :{sys.getsizeof(word2vec)/1024} KB")
    index_to_word = {v: k for k, v in word_to_index.items()}
    return word_to_index, word2vec #, index_to_word


def vectorize_data(data, word_to_index, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        count = 0
        for w in essay:
            count += 1
            if count > sentence_size:
                break
            if w in word_to_index:
                wl.append(word_to_index[w])
            else:
                wl.append(0)
        wl += [0] * ls
        E.append(wl)
    return E



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MANN')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--set_id', type=int, default=1, help="essay set id, 1 <= id <= 8.")
    parser.add_argument('--emb_size', type=int, default=300, help="Embedding size for sentences.")
    parser.add_argument('--token_num', type=int, default=42, help="The number of token in glove (6, 42).")
    parser.add_argument('--feature_size', type=int, default=100, help="Feature size.")
    parser.add_argument('--epochs', type=int, default=200000, help="Number of epochs to train for.")
    parser.add_argument('--test_freq', type=int, default=10, help="Evaluate and print results every x epochs.")
    parser.add_argument('--hops', type=int, default=3, help="Number of hops in the Memory Network.")
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--l2_lambda', type=float, default=0.3, help="Lambda for l2 loss.")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples selected as memories for each score.")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon value for Adam Optimizer.")
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help="Clip gradients to this norm.")
    parser.add_argument('--keep_prob', type=float, default=0.9, help="Keep probability for dropout.")
    args = parser.parse_args()
    bkp=0
    print(args)

    if torch.cuda.is_available():
        print(f"Using GPU:{args.gpu_id}")
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_id)
    else:
        print("!!! Using CPU")
        device = torch.device("cpu")

    timestamp = time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())
    out_file = "./logs/set{}_{}.txt".format(args.set_id, timestamp)
    with open(out_file, 'w', encoding='utf-8') as f:
        for key, value in args.__dict__.items():
            f.write("{}={}".format(key, value))
            f.write("\n")

    # read training, dev and test data
    train_essay_contents, train_essay_scores, train_essay_ids = load_data("train",args.set_id)
    dev_essay_contents, dev_essay_scores, dev_essay_ids = load_data("dev",args.set_id)
    test_essay_contents, test_essay_ids = load_test_data("test",args.set_id)
    min_score = min(train_essay_scores)
    max_score = max(train_essay_scores)
    if args.set_id == 7:
        min_score, max_score = 0, 30
    elif args.set_id == 8:
        min_score, max_score = 0, 60
    score_range = list(range(min_score, max_score + 1))
    # get the vocabulary of training, dev and test datasets.
    all_vocab = all_vocab(train_essay_contents, dev_essay_contents, test_essay_contents)
    print(f"all_vocab len:{len(all_vocab)}")

    # get the length of longest essay in training set
    train_sent_size_list = list(map(len, [content for content in train_essay_contents]))
    max_sent_size = max(train_sent_size_list)
    mean_sent_size = int(np.mean(train_sent_size_list))
    print('max_score={} \t min_score={}'.format(max_score, min_score))
    print('max train sentence size={} \t mean train sentence size={}\n'.format(max_sent_size, mean_sent_size))
    with open(out_file, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('max_score={} \t min_score={}\n'.format(max_score, min_score))
        f.write('max sentence size={} \t mean sentence size={}\n'.format(max_sent_size, mean_sent_size))

    # loading glove. Only select words which appear in vocabulary.
    print("Loading Glove.....")
    t1 = time.time()
    word_to_index, word_to_vec = load_glove(w_vocab=all_vocab, token_num=args.token_num, dim=args.emb_size)
    word_to_vec = np.array(word_to_vec, dtype=np.float32)
    t2 = time.time()
    print(f"Finished loading Glove!, time cost = {(t2-t1):.4f}s\n")

    # [train_essay_size, max_sent_size]  type: list
    train_contents_idx = vectorize_data(train_essay_contents, word_to_index, max_sent_size)
    # [dev_essay_size, max_sent_size]  type: list
    test_contents_idx = vectorize_data(dev_essay_contents, word_to_index, max_sent_size)
    # [test_essay_size, max_sent_size]  type: list
    #test_contents_idx = vectorize_data(test_essay_contents, word_to_index, max_sent_size)

    memory_contents = []
    memory_scores = []
    for i in score_range:
        for j in range(args.num_samples):
            if i in train_essay_scores:
                score_idx = train_essay_scores.index(i)
                score = train_essay_scores.pop(score_idx)  # score=i
                content = train_contents_idx.pop(score_idx)
                memory_contents.append(content)
                memory_scores.append(score)
            else:
                print(f"score {i} is not in train data")

    memory_size = len(memory_contents)  # actual score_range
    train_scores_index = list(map(lambda x: score_range.index(x), train_essay_scores))

    # data size
    n_train = len(train_contents_idx)
    n_dev = len(test_contents_idx)
    n_test = len(test_contents_idx)

    start_list = list(range(0, n_train - args.batch_size, args.batch_size))
    end_list = list(range(args.batch_size, n_train, args.batch_size))
    batches = zip(start_list, end_list)
    batches = [(start, end) for start, end in batches]
    if end_list[len(end_list)-1] != n_train-1:
        batches.append((end_list[len(end_list)-1], n_train-1))

    # model
    model = MANM(word_to_vec=word_to_vec, max_sent_size=max_sent_size, memory_num=memory_size, embedding_size=args.emb_size,
                 feature_size=args.feature_size, score_range=len(score_range), hops=args.hops,
                 l2_lambda=args.l2_lambda, keep_prob=args.keep_prob, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print("----------begin training----------")
    t1 = time.time()
    dev_kappa_result = 0.0
    for ep in range(1, args.epochs+1):
        t2 = time.time()
        total_loss = 0
        np.random.shuffle(batches)
        for start, end in batches:
            contents = np.array(train_contents_idx[start:end], dtype=np.int64)
            scores_index = np.array(train_scores_index[start:end], dtype=np.int64)
            batched_memory_contents = np.array([memory_contents]*(end-start), dtype=np.int64)
            optimizer.zero_grad()
            loss = model(contents, batched_memory_contents, scores_index)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        t3 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        scheduler.step(ep)
        print(f"epoch {ep}/{args.epochs}: all loss={total_loss:.3f}, "
              f"loss/triple={(total_loss/train_essay_contents.__len__()):.6f}, " f"time cost={(t3-t2):.4f}")
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write("epoch {}: total_loss={:.3f}, loss/triple={:.6f}\n".format(ep, total_loss, total_loss/train_essay_contents.__len__()))
        # begin evaluation
        if ep % args.test_freq == 0 or ep == args.epochs:
            print("------------------------------------")
            mid1 = round(n_dev/3)
            mid2 = round(n_dev/3)*2
            dev_batches = [(0, mid1), (mid1, mid2), (mid2, n_dev)]
            all_pred_scores = []
            for start, end in dev_batches:
                test_contents = np.array(test_contents_idx[start:end], dtype=np.int64)
                batched_memory_contents = np.array([memory_contents]*test_contents.shape[0], dtype=np.int64)
                pred_scores = model.test(test_contents, batched_memory_contents).cpu().numpy()
                pred_scores = np.add(pred_scores, min_score)
                all_pred_scores += list(pred_scores)
            
            #print(len(dev_essay_scores))
            #print(dev_essay_scores)
            #print()
            #print(all_pred_scores)
            #print(len(all_pred_scores))
            
            with open(".\metrics\output.txt","a+") as f:
                f.write(str(dev_essay_scores)+"\n")
                f.write(str(all_pred_scores)+"\n")
                
            
            #print(test_contents[0])
            
            dev_kappa_result = kappa(dev_essay_scores, all_pred_scores, weights='quadratic')
            print(f"kappa result={dev_kappa_result}")
            print("------------------------------------")
            with open(out_file, 'a', encoding='utf-8') as f:
               f.write("------------------------------------\n")
               f.write("kappa result={}\n".format(dev_kappa_result))
               f.write("------------------------------------\n")
            if dev_kappa_result>bkp:
               bkp=dev_kappa_result
            model.save_weights(f"C:\\Users\\viksp\\Documents\\Folder_of_Folders\\AES-with-Argumentative-Perturbations\\calling-out-bluff-models_test\\Model5MemoryNetsPytorch\\results\\save_one_{args.set_id}.pth")
            print("------------------------------------")
            print("------------------------------------")
            x= np.array(dev_essay_scores) - np.array(all_pred_scores)
            print(np.mean(x), np.median(x), stats.mode(x))

            print("Original essay scores:")
            print(dev_essay_scores)
            print("Predicted essay scores for adversarial:")
            print(all_pred_scores)
            unique_elements, counts_elements = np.unique(dev_essay_scores, return_counts=True)
            print("Frequency of unique values of the original scores")
            print(np.asarray((unique_elements, counts_elements)))
            unique_elements, counts_elements = np.unique(all_pred_scores, return_counts=True)
            print("Frequency of unique values of the predicted scores")
            print(np.asarray((unique_elements, counts_elements)))            
            unique_elements, counts_elements = np.unique(x, return_counts=True)
            print("Frequency of unique values of the difference (original - adversarial) scores")
            print(np.asarray((unique_elements, counts_elements)))            
            print(f"kappa result={dev_kappa_result}")
            print("------------------------------------")
            print("------------------------------------")
    torch.cuda.empty_cache()