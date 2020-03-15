import csv
import argparse
import numpy as np

def read_gold():
    answers = []
    if args.task == 'A':
        filename = './taskA_answer.csv'
    elif args.task == 'B':
        filename = './taskB_answer.csv'
    else:
        raise ValueError("Wrong task name ({}).".format(args.task))
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            answer = row[1]
            answers.append(answer)
    return answers


def read_prediction():
    predictions = []
    if args.task == 'A':
        filename = './taskA_prediction.csv'
    elif args.task == 'B':
        filename = './taskB_prediction.csv'
    else:
        raise ValueError("Wrong task name ({}).".format(args.task))
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prediction = row[1]
            predictions.append(prediction)
    return predictions


def calculate_accuracy(gold_labels, pred_labels):
    assert len(gold_labels) == len(pred_labels)
    return (np.array(gold_labels) == np.array(pred_labels)).mean()


def main():
    gold_labels = read_gold()
    pred_labels = read_prediction()
    accuracy = calculate_accuracy(gold_labels, pred_labels)
    print(f'Accuracy: {accuracy*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='B', help='which task to evaluate')
    args = parser.parse_args()
    main()