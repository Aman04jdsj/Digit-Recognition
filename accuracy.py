import csv

c = 0
N = 0
with open("test_predictions.csv", "r") as predictions, open("test_label.csv", "r") as labels:
    r1 = csv.reader(predictions)
    r2 = csv.reader(labels)
    for pred, label in zip(r1, r2):
        N += 1
        if int(float(pred[0])) == int(label[0]):
            c += 1

print(f"Accuracy is {c/N}")