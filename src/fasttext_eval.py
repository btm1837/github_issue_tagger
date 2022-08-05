import fasttext
import os
import sys
import os.path
import numpy as np
from numpy import array
from sklearn.model_selection import KFold
import json

"""
From : 
https://github.com/ChristianBirchler/ticket-tagger-analysis/blob/main/code-pipeline/classifiers/classifier.py
"""


if __name__ == '__main__':
    print('* execute ' + sys.argv[0])
    # catch missing arguments
    try:
        a1 = sys.argv[1]
        a2 = sys.argv[2]
    except IndexError as error:
        print('\033[91m' + "Could not read arguments. Please use the correct command format. Example command:")
        print("python classifier.py ../../datasets/data_set-pandas-balanced.txt ./out.txt")
        exit()
    # define paths
    data_set = sys.argv[1]
    f_out = sys.argv[2]
    path_train = os.path.dirname(__file__) + './tmp/tmp_train.txt'
    path_test = os.path.dirname(__file__) + './tmp/tmp_test.txt'

    try:
        print("Converting dataset to array")
        f = open(data_set, 'r+', encoding="UTF-8")
        data = array(f.readlines())
        f.close()

        # array for details
        fold_outputs = []

        # 10 fold loop
        kfold = KFold(10, shuffle=True, random_state=1)
        fold = 1
        for train, test in kfold.split(data):
            print("New tenfold iteration:", str(fold), "-----------------------------------------")
            print("Creating train file")
            tmp_train = open(path_train, "w", encoding="UTF-8")
            for line in data[train]:
                tmp_train.write("".join(line))
            tmp_train.close()

            print("Creating test file")
            tmp_test = open(path_test, "w", encoding="UTF-8")
            for line in data[test]:
                tmp_test.write("".join(line))
            tmp_test.close()

            print("start training...")
            # use this model if you want to enable auto-tuning. Sadly we did not get it to work during our project
            # model = fasttext.train_supervised(input=path_train, autotuneValidationFile=path_test, autotuneDuration=600)
            model = fasttext.train_supervised(input=path_train)  # normal fasttext classification without autotuning
            print("start testing...")
            res = model.test(path_test)
            # get benchmarks
            precision = res[1]
            recall = res[2]
            f1 = 2 * ((precision * recall) / (precision + recall))
            result = {
                '10-Fold iteration:': fold,
                'F1': f1,
                'Recall': recall,
                'Precision': precision
            }
            # log
            print("Fold over, here are results: ")
            print(json.dumps(result, indent=4))
            fold_outputs.append(result)
            fold += 1
        print("Done with 10 fold validation")
        # calculate over-all results
        mean_recall = 0
        mean_precision = 0
        mean_f1 = 0
        for f in fold_outputs:
            mean_f1 += (f['F1'] / 10)
            mean_recall += (f['Recall'] / 10)
            mean_precision += (f['Precision'] / 10)
            # compile results as json
        output = {
            'Results': {
                'F1': mean_f1,
                'Recall': mean_recall,
                'Precision': mean_precision
            },
            'Details': fold_outputs
        }
        dump = json.dumps(output, indent=4)
        print(dump)
        # write to output
        print("Writing output to file")
        o = open(f_out, 'w', encoding="UTF-8")
        o.write(dump)
        o.close()
    except:
        print("An Error occurred")
    # in any case delete existing temporary files
    finally:
        print("Deleting tmp files")
        if os.path.exists(path_train):
            os.remove(path_train)
        if os.path.exists(path_test):
            os.remove(path_test)
        print("Exit.")