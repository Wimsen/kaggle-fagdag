import csv
from model import CNN
import torch
train_comments = []
train_categories = []
index = 0

cutoff_comment_length = 50
with open('data/train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        index += 1
        if index == 1:
            continue
        categories = row[2:]
        categories_array = [int(c) for c in categories]
        comment = row[1].split(" ")
        if len(comment) > cutoff_comment_length:
            comment = comment[:50]
        train_comments.append(comment)
        train_categories.append((categories_array))


all_words = []
for comment in train_comments:
    for word in comment:
        all_words.append(word)

dictionary = sorted(list(set(all_words)))
word_to_index = {w: i for i, w in enumerate(dictionary)}

max_comment_length = max([len(comment) for comment in train_comments])
padding_index = len(dictionary)
train_indices = []

for comment in train_comments:
    non_padded_comment = [word_to_index[word] for word in comment]
    padded_comment = non_padded_comment + [padding_index] * (max_comment_length - len(comment))
    train_indices.append(padded_comment)


model = CNN(dictionary, max_comment_length)

optimizer = torch.optim.Adadelta(model.parameters(), 0.1)
# loss_function = torch.nn.CrossEntropyLoss()

# loss_function = torch.nn.NLLLoss()
loss_function = torch.nn.MultiLabelSoftMarginLoss()
for i in range(100):
    # for (comment, target) in zip(train_indices, train_categories):
    for i in range(0, len(train_indices), 32):
        batch_range = min(32, len(train_indices) - i)
        minibatch = train_indices[i: i + batch_range]
        minibatch_targets = train_categories[i: i + batch_range]
        tensor = torch.LongTensor(minibatch)
        target = torch.Tensor(minibatch_targets)
        predictions = model(tensor)
        # print(predictions.size())
        # print(target.size())

        optimizer.zero_grad()
        loss = loss_function(predictions.squeeze(), target)
        loss.backward()
        optimizer.step()
        print("{} of {} - loss {}".format(i, len(train_indices), loss))
    print("epoch {} of {}, loss {}".format(i, 100, loss))

