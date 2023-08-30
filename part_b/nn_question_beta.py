import sys
sys.path.append("../")
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch

import math
from torch import sigmoid
from part_a import item_response


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
We can optionally pass in the beta value which is added to the latent vector. 
'''
class AutoEncoder(nn.Module):
    def __init__(self, num_students, k=100, beta_latent_dim=1):
        """ Initialize a class AutoEncoder.

        :param num_students: int
        :param k: int
        :param beta_latent_dim: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, k)
        self.h = nn.Linear(k + beta_latent_dim, num_students)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs, beta=None):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        question_raw_latent = F.sigmoid(self.g(inputs))

        if beta is not None:
            beta_tensor = torch.tensor([[beta]], dtype=torch.float32)
            question_latent = torch.cat(
                (question_raw_latent, beta_tensor), dim=-1)
            # Concatenate beta to the latent representation
        else:
            question_latent = question_raw_latent

        decoded = F.sigmoid(self.h(question_latent))
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, betas):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param betas: list
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

            if betas is not None:
                beta = betas[question_id]
                output = model(inputs, beta)
            else:
                output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[question_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + \
                (lamb / 2) * (model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, betas)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

    return model, train_losses, val_accuracies


def evaluate(model, train_data, valid_data, betas):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param betas: list
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[q]).unsqueeze(0)
        if betas is not None:
            output = model(inputs, betas[q])
        else:
            output = model(inputs)

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, question_train_matrix, valid_data, test_data = load_data()
    train_data = load_train_csv("../data")

    if torch.cuda.is_available():
        print('using GPU')
        device = torch.device('cuda')
    else:
        device = 'cpu'

    _, betas, _, _ = item_response.irt(
        train_data=train_data, val_data=valid_data, lr=0.004, iterations=160)

    # Set model hyperparameters.
    # k_values = [10, 50, 100, 200, 500]
    k_values = [10]

    # learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    learning_rates = [0.01]

    # epochs = [2, 5, 10]
    epochs = [10]  # 3, 5, 10, 15

    # lambdas = [0.001, 0.01, 0.1, 1]
    lamb = 0.001

    test_accuracy_list = []
    max_accuracy = 0
    optimal_parameters = []
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []

    for k in k_values:
        for lr in learning_rates:
            for num_epoch in epochs:
                model = AutoEncoder(num_students=question_train_matrix.shape[1], k=k,
                            beta_latent_dim=1)

                model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix,
                                                   zero_train_matrix, valid_data, num_epoch, betas)

                valid_acc = val_acc[-1]

                if valid_acc > max_accuracy:
                    max_accuracy = valid_acc
                    optimal_parameters = [k, lr, num_epoch]
                    optimal_val_accuracies = val_acc
                    optimal_train_losses = train_loss
                    optimal_model = model

                print(f'k: {k}, Learning Rate: {lr}, Epoch: {num_epoch}')
                print(f'Validation Accuracy: {valid_acc}')
                print(f'===============================================')

            print(f'Best k: {optimal_parameters[0]}, Best Learning Rate: {optimal_parameters[1]}'
                  f', Epoch: {optimal_parameters[2]}')
            print(f'Best Validation Accuracy: {max_accuracy}')

    test_accuracy = evaluate(optimal_model, zero_train_matrix, test_data, betas)
    print(f'Test Accuracy: {test_accuracy}')



if __name__ == "__main__":
    main()
