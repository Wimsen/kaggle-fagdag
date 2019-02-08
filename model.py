import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, dictionary, max_comment_length):
        super(CNN, self).__init__()
        self.word_dim = 15
        self.output_features = 20


        self.embedding = nn.Embedding(len(dictionary) + 1, self.word_dim, padding_idx=len(dictionary))
        self.max_comment_length = max_comment_length

        self.conv_1 = nn.Conv1d(1, self.output_features, 3 * self.word_dim, self.word_dim)
        self.conv_2 = nn.Conv1d(1, self.output_features, 4 * self.word_dim, self.word_dim)
        # self.conv_3 = nn.Conv1d(1, self.output_features, 5 * self.word_dim, self.word_dim)

        self.fc = nn.Linear(2 * self.output_features, 6)
        self.softmax = nn.Softmax()


    def forward(self, input):

        # 32 * setning_lengde

        input = self.embedding(input).view(-1, 1, self.word_dim * self.max_comment_length)
        # 32 * 1 * (self.word_dim * antall_ord)

        conv_results_1 = F.max_pool1d(F.relu(self.conv_1(input)), self.max_comment_length - 3 + 1).view(-1, self.output_features)
        conv_results_2 = F.max_pool1d(F.relu(self.conv_2(input)), self.max_comment_length - 4 + 1).view(-1, self.output_features)
        # conv_results_3 = F.max_pool1d(F.relu(self.conv_3(input)), self.max_comment_length - 5 + 1).view(-1, 20)
        # conv_results = torch.cat([conv_results_1, conv_results_2, conv_results_3], 1)
        conv_results = torch.cat([conv_results_1, conv_results_2], 1)
        output = self.fc(conv_results)
        output = self.softmax(output)
        return output

