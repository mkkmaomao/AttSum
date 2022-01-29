import nltk
import numpy as np


def main():
    with open("./body.train.txt", encoding="utf8") as fbody, open("./title.train.txt",encoding="utf8") as ftitle:
        test_body_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in fbody]
        test_title_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in ftitle]

        print(test_body_length[0])
        print(f'median_train_body_length: {np.median(test_body_length)}')
        print(f'mean_train_body_length: {np.mean(test_body_length)}')
        print(f'max_train_body_length: {max(test_body_length)}')
        print(f'min_train_body_length: {min(test_body_length)}')

        print(f'median_train_title_length: {np.median(test_title_length)}')
        print(f'mean_train_title_length: {np.mean(test_title_length)}')
        print(f'max_train_title_length: {max(test_title_length)}')
        print(f'min_train_title_length: {min(test_title_length)}')

        train_body = [nltk.word_tokenize(line1) for line1 in fbody]
        train_title = [nltk.word_tokenize(line) for line in fbody]
        print(train_body)
        print(train_title)
    fbody.close()
    ftitle.close()


    with open("body.valid.txt", encoding="utf8") as fbody, open("title.valid.txt", encoding="utf8") as ftitle:
        test_body_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in fbody]
        test_title_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in ftitle]

        print(f'median_valid_body_length: {np.median(test_body_length)}')
        print(f'mean_valid_body_length: {np.mean(test_body_length)}')
        print(f'max_valid_body_length: {max(test_body_length)}')
        print(f'min_valid_body_length: {min(test_body_length)}')

        print(f'median_valid_title_length: {np.median(test_title_length)}')
        print(f'mean_valid_title_length: {np.mean(test_title_length)}')
        print(f'max_valid_title_length: {max(test_title_length)}')
        print(f'min_valid_title_length: {min(test_title_length)}')
    fbody.close()
    ftitle.close()


    with open("body.test.txt", encoding="utf8") as fbody, open("title.test.txt", encoding="utf8") as ftitle:
        test_body_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in fbody]
        test_title_length = [len(nltk.word_tokenize(line, preserve_line=False)) for line in ftitle]

        print(f'median_test_body_length: {np.median(test_body_length)}')
        print(f'mean_test_body_length: {np.mean(test_body_length)}')
        print(f'max_test_body_length: {max(test_body_length)}')
        print(f'min_test_body_length: {min(test_body_length)}')

        print(f'median_test_title_length: {np.median(test_title_length)}')
        print(f'mean_test_title_length: {np.mean(test_title_length)}')
        print(f'max_test_title_length: {max(test_title_length)}')
        print(f'min_test_title_length: {min(test_title_length)}')
    fbody.close()
    ftitle.close()

def read(file):
    import re
    with open(file,"r",encoding="utf8") as f:
        list = []
        for line in f.readlines():
            line = line.strip("\n")
            list.append(line)

        print(f'======={file}=======')
        print('The number of data:',len(list))

        word = [nltk.word_tokenize(sentence) for sentence in list]
        count_word = []
        for sen in range(len(word)):
            a = [x for x in word[sen] if re.match("\S*[A-Za-z0-9]+\S*", x)]
            count_word.append(a)
            # print('sen:',word[sen])
            # print('processed sen:',a)
        max_word = max(word, key=len, default='')
        max_word_length = len(max(word, key=len, default=''))
        min_word_length = len(min(word, key=len, default=''))

        max_count_word = max(count_word, key=len, default='')
        max_count_word_length = len(max(count_word, key=len, default=''))
        min_count_word_length = len(min(count_word, key=len, default=''))

        print('max word :', max_word)
        print('max word length:',max_word_length)
        print('min word length:', min_word_length)

        print('max count_word :', max_count_word)
        print('max count_word length:',max_count_word_length)
        print('min count_word length:', min_count_word_length)

def read_json(json_file):
    import json
    import re
    with open(json_file) as f:
        all_issues = json.load(f)

    list = []
    max_length = 0
    for idx, issue in enumerate(all_issues):
        list.extend(issue)
    print("the total number of issues is ", len(list))
    print('the title of the first issue is ',list[0]['title'])

    for i in range(len(list)):
        list[i]['title'] = nltk.word_tokenize(list[i]['title'])
        list[i]['title'] = [x for x in list[i]['title'] if re.match("\S*[A-Za-z0-9]+\S*", x)]
        if (len(list[i]['title'])>max_length):
            max_length = len(list[i]['title'])
        if (len(list[i]['title'])<max_length):
            print('max length of titles', max_length)


if __name__ == "__main__":
    # main()
    train_body_path = r'./body.train.txt'
    train_title_path = r'./title.train.txt'
    valid_body_path = r'./body.valid.txt'
    valid_title_path = r'./title.valid.txt'
    test_body_path = r'./body.test.txt'
    test_title_path = r'./title.test.txt'
    json_file = r'./refined_333563issues.json'

    # read(train_body_path)
    # read(train_title_path)
    # read(valid_body_path)
    # read(valid_title_path)
    # read(test_body_path)
    # read(test_title_path)
    read_json(json_file)