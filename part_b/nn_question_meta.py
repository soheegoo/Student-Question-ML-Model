import sys
sys.path.append("../")
from torch import sigmoid
import math
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from utils import *



def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, question_train_matrix, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        question_train_matrix: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
    """
    # Transpose the matrix so each question is a row and columns are students
    question_train_matrix = load_transposed_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = question_train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(question_train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    question_train_matrix = torch.FloatTensor(question_train_matrix)
    return zero_train_matrix, question_train_matrix, valid_data, test_data


'''
This is a question based autoencoder instead of a student based autoencoder which takes
in question vectors as inputs.
We can optionally pass in the meta data values which is added to the latent vector. 
'''
class AutoEncoder(nn.Module):
    def __init__(self, num_students, num_subjects, subject_latent_dim=5, k=100):
        """ Initialize a class AutoEncoder.

        :param num_students: int
        :param num_subjects: int
        :param subject_latent_dim: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, k)
        self.h = nn.Linear(k + subject_latent_dim,
                           num_students)
        # adding dimension for encoded meta data vector

        self.subject_enc_linear = nn.Linear(num_subjects, subject_latent_dim)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs, meta_data=None):
        """ Return a forward pass given inputs.

        :param inputs: question 1D FloatTensor
        :param meta_data: meta data 1D FloatTensor
        :return: reconstructed question 1D FloatTensor
        """
        question_latent = sigmoid(self.g(inputs))

        if meta_data is not None:
            subject_latent = torch.sigmoid(self.subject_enc_linear(meta_data))
            combined_latent = torch.cat(
                (question_latent, subject_latent), dim=-1)
            decoded = sigmoid(self.h(combined_latent))
        else:
            decoded = sigmoid(self.h(question_latent))
        return decoded


def train(model,lr,lamb,train_data,zero_train_data,valid_data,num_epoch,metas=None):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param metas: 2D FloatTensor
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = train_data.shape[0]

    train_losses = []
    val_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for question_id in range(num_question):
            inputs = Variable(zero_train_data[question_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()

            if metas is None:
                output = model(inputs, meta_data=None)
            else:
                # get meta data of current question, pass it to model
                meta = metas[question_id].unsqueeze(0)
                output = model(inputs, meta_data=meta)

            # Mask the target to replace missing values with the corresponding values
            # from the output
            nan_mask = np.isnan(train_data[question_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + \
                (lamb / 2) * (model.get_weight_norm())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, metas)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

    return model, train_losses, val_accuracies


def evaluate(model, train_data, valid_data, metas=None):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param metas: 2D FloatTensor
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[q].unsqueeze(0))
        
        if metas is None:
            output = model(inputs, meta_data=metas)
        else:
            # get meta data of current question, pass it to model
            meta = metas[q].unsqueeze(0)
            output = model(inputs, meta_data=meta)

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def get_metadata(meta_file, num_questions, num_subjects):
    """
    Read the metadata csv file and convert into a metadata matrix.
    The matrix has shape (num_questions, num_subjects).
    matrix[x, y] is 1 if question x is of subject y and 0 otherwise
    """
    meta_from_csv = pd.read_csv(meta_file).to_numpy()
    # initialize a matrix of all zeros
    meta_data_matrix = np.zeros((num_questions, num_subjects))
    for question, subject in meta_from_csv:
        # splitting the string of subjects into a list of subjects
        subject_list = subject[1:-1].split(',')
        # converting subject strings into subject numbers
        subject_list = [int(x) for x in subject_list]
        for subject_id in subject_list:
            meta_data_matrix[question, subject_id] = 1
    return meta_data_matrix


def main():
    zero_train_matrix, question_train_matrix, valid_data, test_data = load_data()

    if torch.cuda.is_available():
        print('using GPU')
        device = torch.device('cuda')
    else:
        device = 'cpu'

    question_csv_data = load_train_csv("../data")
    subject_csv_data = pd.read_csv("../data/subject_meta.csv")
    num_question = max(question_csv_data['question_id']) + 1
    num_subject = max(subject_csv_data['subject_id']) + 1
    meta_data_matrix = get_metadata(
        '../data/question_meta.csv', num_question, num_subject)
    meta_data = torch.FloatTensor(meta_data_matrix)

    # Set model hyperparameters.
    # meta_latent_dim_list = [2, 3, 4, 5]
    meta_latent_dim_list = [5]

    # k_values = [10, 50, 100, 200, 500]
    k_values = [500]

    # learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    learning_rates = [0.01]

    # epochs = [2, 5, 10]
    epochs = [10]

    # lambdas = [0.001, 0.01, 0.1, 1]
    lamb = 0.001

    max_accuracy = 0
    optimal_parameters = []
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []

    for meta_latent_dim in meta_latent_dim_list:
        for k in k_values:
            for lr in learning_rates:
                for num_epoch in epochs:
                    model = AutoEncoder(num_students=question_train_matrix.shape[1],
                                        num_subjects=num_subject,
                                        subject_latent_dim=meta_latent_dim, k=k)
                    model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix,
                                                       zero_train_matrix, valid_data, num_epoch,
                                                       metas=meta_data)

                    valid_acc = val_acc[-1]

                    if valid_acc > max_accuracy:
                        max_accuracy = valid_acc
                        optimal_parameters = [
                            k, lr, num_epoch, meta_latent_dim]
                        optimal_val_accuracies = val_acc
                        optimal_train_losses = train_loss
                        optimal_model = model

                    print(f'k: {k}, Learning Rate: {lr}, Epoch: {num_epoch}, Meta Latent Dim: {meta_latent_dim}')
                    print(f'Validation Accuracy: {valid_acc}')
                    print(f'===============================================')

    print(f'Best k: {optimal_parameters[0]}, Best Learning Rate: {optimal_parameters[1]}'
          f', Epoch: {optimal_parameters[2]}, Meta Latent Dim: {optimal_parameters[3]}')
    print(f'Best Validation Accuracy: {max_accuracy}')

    test_accuracy = evaluate(
        optimal_model, zero_train_matrix, test_data, metas=meta_data)

    print(f'Test Accuracy: {test_accuracy}')



if __name__ == "__main__":
    main()
