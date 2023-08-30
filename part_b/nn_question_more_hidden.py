import sys
sys.path.append("../")
from part_a import item_response
from torch import sigmoid
import math
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
We can optionally add 2 more hidden layers to make it a 5 layer neural net architecture.
'''
class AutoEncoder(nn.Module):
    def __init__(self, num_students, hidden_l, k=100, j=10):
        """ Initialize a class AutoEncoder.

        :param num_students: int
        :param hidden_l: boolean
        :param k: int
        :param j: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        # Adding two more hidden layers to help with underfitting
        if hidden_l is False:
            self.hidden_l = False
            self.g1 = nn.Linear(num_students, k)
            self.h1 = nn.Linear(k, num_students)
        else:
            self.hidden_l = True
            self.g1 = nn.Linear(num_students, k)
            self.g2 = nn.Linear(k, j)
            self.h2 = nn.Linear(j, k)
            self.h1 = nn.Linear(k, num_students)

    def get_weight_norm(self, hidden_l=True):
        """ Return ||W^1||^2 + ||W^2||^2 + ||W^3||^3 + ||W^4||^4

        :return: float
        """
        g2_w_norm, h2_w_norm = 0, 0
        if self.hidden_l is not False:
            g2_w_norm = torch.norm(self.g2.weight, 2) ** 2
            h2_w_norm = torch.norm(self.h2.weight, 2) ** 2
        g1_w_norm = torch.norm(self.g1.weight, 2) ** 2
        h1_w_norm = torch.norm(self.h1.weight, 2) ** 2

        return g1_w_norm + g2_w_norm + h1_w_norm + h2_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: question vector.
        :return: reconstructed question vector.
        """
        question_latent = F.sigmoid(self.g1(inputs))

        if self.hidden_l is not False:
            question_latent = F.sigmoid(self.g2(question_latent))
            decoded = F.sigmoid(self.h2(question_latent))
            decoded = F.sigmoid(self.h1(decoded))
        else:
            decoded = F.sigmoid(self.h1(question_latent))

        return decoded


def train(model, lr, lamb, question_train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param question_train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: (model, train_losses, val_accuracies)
        WHERE:
        model: trained autoencoder
        train_losses: list
        val_accuracies: list
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = question_train_data.shape[0]

    train_losses = []
    val_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for question_id in range(num_question):
            # takes the entire row for the specific question, makes it a 2D tensor
            inputs = Variable(zero_train_data[question_id]).unsqueeze(0)

            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to replace missing values with the corresponding values
            # from the output
            nan_mask = np.isnan(
                question_train_data[question_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + \
                (lamb / 2) * (model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

    return model, train_losses, val_accuracies


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[q].unsqueeze(0))
        output = model(inputs)

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, question_train_matrix, valid_data, test_data = load_data()

    if torch.cuda.is_available():
        print('using GPU')
        device = torch.device('cuda')
    else:
        device = 'cpu'

    """
    Finding the optimal Hyperparameter Values. 

    # Set model hyperparameters.
    k_values = [10, 50, 100, 200, 500]

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    epochs = [2, 5, 10]

    test_accuracy_list = []

    max_accuracy = 0
    optimal_parameters = []
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []

    for k in k_values:
        for lr in learning_rates:
            for num_epoch in epochs:
                model = AutoEncoder(num_students=question_train_matrix.shape[1], k=k)
                model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix,
                                                   zero_train_matrix, valid_data, num_epoch)

                valid_accuracy = val_acc[-1]

                if valid_accuracy > max_accuracy:
                    max_accuracy = valid_accuracy
                    optimal_parameters = [k, lr, num_epoch]
                    optimal_val_accuracies = val_acc
                    optimal_train_losses = train_loss
                    optimal_model = model

                print(f'k: {k}, Learning Rate: {lr}, Epoch: {num_epoch}')
                print(f'Validation Accuracy: {valid_accuracy}')
                print(f'===============================================')

    print(f'Best k: {optimal_parameters[0]}, Best Learning Rate: {optimal_parameters[1]}'
          f', Epoch: {optimal_parameters[2]}')
    print(f'Best Validation Accuracy: {max_accuracy}')

    #Finding the best lamda hyperparameter values

    lambdas = [0.001, 0.01, 0.1, 1]

    max_accuracy = 0
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []
    optimal_lambda = 0

    for lamb in lambdas:
        model = AutoEncoder(num_students=question_train_matrix.shape[1], k=k_star)
        model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix, zero_train_matrix,
                                           valid_data, num_epoch)

        valid_acc = val_acc[-1]

        print(f' Lambda: {lamb}, Validation Accuracy: {valid_acc}')

        if valid_acc > max_accuracy:
            max_accuracy = valid_acc
            optimal_lambda = lamb
            optimal_train_losses = train_loss
            optimal_val_accuracies = val_acc
            optimal_model = model

    print(f'Best lambda: optimal_lambda}}')
    """

    # Training the model with the best hyperparameters
    k_star = 50
    lr = 0.005
    num_epoch = 10
    lamb = 0.001
    js = [5, 10, 15, 20]
    train_losses = []
    val_accs = []

    max_accuracy = 0
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []
    optimal_lambda = 0

    for j in js:
        model = AutoEncoder(
            num_students=question_train_matrix.shape[1], hidden_l=True, k=k_star, j=j)
        model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix, zero_train_matrix,
                                           valid_data, num_epoch)
        valid_accuracy = val_acc[-1]

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            optimal_j = j
            optimal_train_losses = train_loss
            optimal_val_accuracies = val_acc
            optimal_model = model

        print(f'k: {k_star}, Learning Rate: {lr}, Epoch: {num_epoch}, j: {j}')
        print(f'Validation Accuracy: {valid_accuracy}')
        print(f'===============================================')

    test_accuracy = evaluate(optimal_model, zero_train_matrix, test_data)

    print(f'Test Accuracy: {test_accuracy}')
    print("Best j hyperparameter value: " + str(optimal_j))

if __name__ == "__main__":
    main()
