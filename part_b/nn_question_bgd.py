import sys
sys.path.append("../")
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
'''
class AutoEncoder(nn.Module):
    def __init__(self, num_students, k):
        """ Initialize a class AutoEncoder.

        :param num_students: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, k)
        self.h = nn.Linear(k, num_students)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: question vector.
        :return: question vector.
        """
        question_latent = sigmoid(self.g(inputs))

        decoded = sigmoid(self.h(question_latent))
        return decoded


def train(model, lr, lamb, question_train_data, zero_train_data, valid_data, num_epoch, batch_size):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param question_train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param batch_size: int
    :return: (model, train_losses, val_accuracies)
        WHERE:
        model: trained autoencoder
        train_losses: list 
        val_accuracies: list
    """
    # Tell PyTorch you are training the model.
    model.train()

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(zero_train_data, question_train_data),
                                              batch_size=batch_size, shuffle=True)

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = question_train_data.shape[0]

    train_losses = []
    val_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for batch_zero_train_data, batch_train_data in data_loader:

            optimizer.zero_grad()
            batch_output = model(batch_zero_train_data)

            # Mask the target to replace missing values with the corresponding values
            # from the output
            nan_mask = torch.isnan(batch_train_data)
            batch_target = batch_train_data.clone()
            batch_target[nan_mask] = batch_output[nan_mask]

            loss = torch.sum((batch_output - batch_target) ** 2.) + \
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

    
    # Set model hyperparameters.
    k_star = 100
    lr = 0.01
    num_epoch = 10
    lamb = 0.001

    # Finding the optimal Hyperparameter Values for batch size
    batches = [2, 5, 8, 10, 15, 20]

    max_accuracy = 0
    optimal_parameters = []
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []

    for batch_size in batches:
        model = AutoEncoder(
            num_students=question_train_matrix.shape[1], k=k_star)
        model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix,
                                           zero_train_matrix, valid_data, num_epoch, batch_size)

        valid_accuracy = val_acc[-1]

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            optimal_parameters = [k_star, lr, num_epoch, batch_size]
            optimal_val_accuracies = val_acc
            optimal_train_losses = train_loss
            optimal_model = model

            print(
                f'k: {k_star}, Learning Rate: {lr}, Epoch: {num_epoch}, Batchsize: {batch_size}')
            print(f'Validation Accuracy: {valid_accuracy}')
            print(f'===============================================')

    print(f'Best k: {optimal_parameters[0]}, Best Learning Rate: {optimal_parameters[1]}'
          f', Epoch: {optimal_parameters[2]}, Best Batch Size: {optimal_parameters[3]}')
    print(f'Best Validation Accuracy: {max_accuracy}')

    # Model is trained with optimal hyperparameters for k, lamda, epoch number, learning rate
    test_accuracy = evaluate(optimal_model, zero_train_matrix, test_data)

    print(f'Test Accuracy: {test_accuracy}')


if __name__ == "__main__":
    main()
