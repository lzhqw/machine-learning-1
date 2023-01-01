from Bayes import NaiveBayes
import re
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def load_data(data_path):
    X = []
    y = []
    with open(data_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                label, sentence = line.split('\t')
                sentence = sentence.lower()
                y.append(1 if label == 'spam' else 0)
                split = re.split('[\n.!?,\t ]', sentence)
                sentence = []
                for word in split:
                    if word != '':
                        sentence.append(word)
                X.append(sentence)

            else:
                break
    return X, y



def get_vocab(data):
    vocab = {}
    for sentence in data:
        for word in sentence:
            vocab.setdefault(word, 0)
            vocab[word] += 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    new_vocab = {}
    for i, item in enumerate(vocab):
        new_vocab[item[0]] = i

    return new_vocab

def word2id(data, vocab):
    for i in tqdm(range(len(data))):
        vector = [0 for i in range(len(vocab))]
        for word in data[i]:
            vector[vocab[word]] += 1
        data[i] = vector
    return data

if __name__ == '__main__':
    data_path = 'SMSSpamCollection.txt'

    X, y = load_data(data_path)
    print(X)
    vocab = get_vocab(X)
    print(vocab)
    X = word2id(X, vocab)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    model = NaiveBayes()
    model.train(X,y)

    true = 0
    predict = []
    for i in range(X.shape[0]):
        predict.append(model.predict(X[i,:]))
    print(accuracy_score(y, predict))
    print(classification_report(y,predict))
