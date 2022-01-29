import json

import nltk


def word_count(file_name):
    import collections
    word_freq = collections.defaultdict(int)
    with open(file_name, encoding="utf-8") as f:
        for l in f:
            for w in l.strip().split():
                word_freq[w] += 1
    return word_freq

def build_dict(file_name, min_word_freq):
    word_freq = word_count(file_name)
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items()) 
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*word_freq_sorted))
    freq = []
    for i in range(len(word_freq_sorted)):
        freq.append(word_freq_sorted[i][1])
    word_freq_dict = dict(zip(words, freq))
    print('word_freq_sorted', word_freq_dict)
    word_idx = dict(zip(words, range(len(words))))
    # word_idx['<unk>'] = len(words) #unk
    # return word_idx
    return word_freq_dict

def read():
    with open("./new_data/refined_333563issues.json",encoding="utf-8") as f:
        valid_issues_train, valid_issues_val, valid_issues_test = json.load(f)
    valid_issues = valid_issues_train + valid_issues_val + valid_issues_test
    for idx, issue in enumerate(valid_issues):
        issue['body'] = " ".join(nltk.word_tokenize(issue['body'], preserve_line=False)).strip().lower()
        issue['title'] = " ".join(nltk.word_tokenize(issue['title'], preserve_line=False)).strip().lower()

    with open("forVocab_dataset_freq.txt", "w", encoding="utf-8") as fbody:
        bodies = [x['body'] + "\n" for x in valid_issues]
        fbody.writelines(bodies)

def writetofile(dict):

    fileObject = open('vocabulary_dataset_freq.txt', 'w',encoding="utf-8")
    # print(dict)
    num = 0
    for key, value in dict.items():
        if(num < 50000):
            fileObject.write('%s %s\n' % (key, value))
        num = num+1
    fileObject.close()

if __name__ == "__main__":
    read()
    dict = build_dict("forVocab_dataset_freq.txt", 6)
    writetofile(dict)