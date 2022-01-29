import json
import re

import nltk
import numpy as np


def main():
    with open("./refined_333563issues_versionsolved_sub.json") as f:
        valid_issues_train, valid_issues_val, valid_issues_test = json.load(f)

    train_body_length = []
    train_title_length = []

    val_body_length = []
    val_title_length = []

    test_body_length = []
    test_title_length = []

    for idx, issue in enumerate(valid_issues_train):
        if idx % 10000 == 0:
            print("current idx:", idx, "/", len(valid_issues_train))

        issue_body_tokenize = nltk.word_tokenize(issue['body'])
        issue_title_tokenize = nltk.word_tokenize(issue['title'])
        issue_title_words = [x.lower() for x in issue_title_tokenize if re.match("\S*[A-Za-z0-9]+\S*", x)]
        train_body_length.append(len(issue_body_tokenize))
        train_title_length.append(len(issue_title_words))

    print(f'median_train_body_length: {np.median(train_body_length)}')
    print(f'mean_train_body_length: {np.mean(train_body_length)}')
    print(f'max_train_body_length: {max(train_body_length)}')
    print(f'min_train_body_length: {min(train_body_length)}')
    print(f'median_train_title_length: {np.median(train_title_length)}')
    print(f'mean_train_title_length: {np.mean(train_title_length)}')
    print(f'max_train_title_length: {max(train_title_length)}')
    print(f'min_train_title_length: {min(train_title_length)}')


    for idx, issue in enumerate(valid_issues_val):
        if idx % 10000 == 0:
            print("current idx:", idx, "/", len(valid_issues_val))

        issue_body_tokenize = nltk.word_tokenize(issue['body'])
        issue_title_tokenize = nltk.word_tokenize(issue['title'])
        issue_title_words = [x.lower() for x in issue_title_tokenize if re.match("\S*[A-Za-z0-9]+\S*", x)]
        val_body_length.append(len(issue_body_tokenize))
        val_title_length.append(len(issue_title_words))


    print(f'median_valid_body_length: {np.median(val_body_length)}')
    print(f'mean_valid_body_length: {np.mean(val_body_length)}')
    print(f'max_valid_body_length: {max(val_body_length)}')
    print(f'min_valid_body_length: {min(val_body_length)}')
    print(f'median_valid_title_length: {np.median(val_title_length)}')
    print(f'mean_valid_title_length: {np.mean(val_title_length)}')
    print(f'max_valid_title_length: {max(val_title_length)}')
    print(f'min_valid_title_length: {min(val_title_length)}')


    for idx, issue in enumerate(valid_issues_test):
        if idx % 10000 == 0:
            print("current idx:", idx, "/", len(valid_issues_test))

        issue_body_tokenize = nltk.word_tokenize(issue['body'])
        issue_title_tokenize = nltk.word_tokenize(issue['title'])
        issue_title_words = [x.lower() for x in issue_title_tokenize if re.match("\S*[A-Za-z0-9]+\S*", x)]
        test_body_length.append(len(issue_body_tokenize))
        test_title_length.append(len(issue_title_words))


    print(f'median_test_body_length: {np.median(test_body_length)}')
    print(f'mean_test_body_length: {np.mean(test_body_length)}')
    print(f'max_test_body_length: {max(test_body_length)}')
    print(f'min_test_body_length: {min(test_body_length)}')
    print(f'median_test_title_length: {np.median(test_title_length)}')
    print(f'mean_test_title_length: {np.mean(test_title_length)}')
    print(f'max_test_title_length: {max(test_title_length)}')
    print(f'min_test_title_length: {min(test_title_length)}')

    f.close()

def seperate_jsonfiles(train, val, test):
    with open("./refined_333563issues_versionsolved_sub.json") as f:
        valid_issues_train, valid_issues_val, valid_issues_test = json.load(f)
    with open(train,'w') as f:
        json.dump(valid_issues_train,f)
    with open(val,'w') as f:
        json.dump(valid_issues_val,f)
    with open(test,'w') as f:
        json.dump(valid_issues_test,f)



def main_solveversion(solution, jsonfile):
    assert solution in ['none', 'tag', 'sub']
    with open(jsonfile) as f:
        valid_issues_train, valid_issues_val, valid_issues_test = json.load(f)

    valid_issues = valid_issues_train + valid_issues_val + valid_issues_test
    for idx, issue in enumerate(valid_issues):
        if idx % 50000 == 0:
            print("current idx :", idx, "/", len(valid_issues))

        version_list = issue["_spctok"]["ver"]
        for version, stat in sorted(version_list.items(), key=lambda x: (len(x[0]))):

            if solution == 'sub':  # substitute with veridID, ID: appear order
                issue['body'] = re.sub(re.escape(version), " verid" + str(stat[0] + 1) + " ", issue['body'],
                                       flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(version), " verid" + str(stat[0] + 1) + " ", issue['title'],
                                        flags=re.IGNORECASE)

        # final lowercase transformation & tokenize
        issue['body'] = " ".join(nltk.word_tokenize(issue['body'], preserve_line=False)).strip().lower()
        issue['title'] = " ".join(nltk.word_tokenize(issue['title'], preserve_line=False)).strip().lower()


    with open("refined_333563issues_versionsolved.json", 'w') as f:
        json.dump(valid_issues, f)

if __name__ == "__main__":
    main()
    # seperate_jsonfiles("./refined_sub_train.json","./refined_sub_val.json","./refined_sub_test.json")
    # main_solveversion('sub', "refined_333563issues.json")