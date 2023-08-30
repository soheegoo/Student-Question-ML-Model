import sys
sys.path.append("../")
from utils import *
from torch.autograd import Variable
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)  # W1
        self.h = nn.Linear(k, num_question)  # W2

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = F.sigmoid(self.g(out))  # going through encoder

        out = F.sigmoid(self.h(out))  # going through decoder

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out  # reconstructed vector


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module, instance of an Autoencoder NN
    :param lr: float
    :param lamb: float, regularization parameter
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    num_student = train_data.shape[0]

    train_losses = []  # going to add the training loss after each epoch
    val_accuracies = []  # going to add the validation accuracy after each epoch

    for epoch in range(0, num_epoch): 
        train_loss = 0.

        for user_id in range(num_student): 
            # inputs is a tensor with shape (1, num_features) for one user
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)

            target = inputs.clone()

            optimizer.zero_grad() 
            output = model(inputs) 

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())

            # replaces the missing values in the target tensor with corresponding predicted values
            # from the output tensor
            target[0][nan_mask] = output[0][nan_mask]

            # computes the loss for one input
            loss = torch.sum((output - target) ** 2.) + (lamb / 2) * (model.get_weight_norm())

            # performs backpropagation 
            loss.backward()

            # adds value of current batch's loss to running total of training loss
            train_loss += loss.item()

            # updates the models param using gradients calculated through backprop
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Lamb: {}\t "
              "Valid Acc: {}".format(epoch, train_loss, lamb, valid_acc))

        # adds training loss and validation accuracy after one epoch
        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

    return model, train_losses, val_accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    if torch.cuda.is_available():
        print('using GPU')
        device = torch.device('cuda')
    else:
        device = 'cpu'

    # Set model hyperparameters.
    #k_values = [10, 50, 100, 200, 500] 
    k_values = [10]

    #learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    learning_rates = [0.1]

    #epochs = [2, 5, 10]
    epochs = [10]

    lamb = 0  # Unused, no regularization for this part, will automatically make regularizer zero

    # Set optimization hyperparameters.
    k_star = 0
    k_star_model = None
    k_star_train_losses = None
    k_star_val_acc = None
    lr_star = 0
    epoch_star = 0
    max_accuracy = 0

    for k in k_values:
        for lr in learning_rates:
            for num_epoch in epochs:
                model = AutoEncoder(num_question=train_matrix.shape[1],
                                    k=k) 
                model, train_loss, val_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                                   valid_data, num_epoch)

                valid_acc = val_acc[-1]

                if valid_acc > max_accuracy:
                    k_star = k
                    k_star_model = model
                    k_star_val_acc = val_acc
                    k_star_train_losses = train_loss
                    lr_star = lr
                    epoch_star = num_epoch
                    max_accuracy = valid_acc

    print(f"Best k: {k_star}, Best validation accuracy: {max_accuracy}")
    print(f"Best Learning Rate: {lr_star}, Best epoch: {epoch_star}")

    # Calculate final test accuracy part d)
    test_accuracy = evaluate(k_star_model, zero_train_matrix, test_data)
    print(f'Best k Final Test Accuracy: {test_accuracy}')

    num_epoch = epoch_star

    epochs = range(num_epoch)

    plt.plot(epochs, k_star_train_losses)
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()

    plt.plot(epochs, k_star_val_acc)
    plt.title('Validation Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.show()

    # e)

    regularization_penalties = [0.001, 0.01, 0.1, 1]
    max_accuracy = 0
    optimal_lambda = 0
    optimal_train_losses = []
    optimal_val_accuracies = []
    trained_model = None

    for lamb in regularization_penalties:
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k_star)
        model, train_loss, val_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                           valid_data, num_epoch)

        valid_acc = val_acc[-1]

        if valid_acc > max_accuracy:
            max_accuracy = valid_acc
            optimal_lambda = lamb
            optimal_train_losses = train_loss
            optimal_val_accuracies = val_acc
            trained_model = model

    test_accuracy = evaluate(trained_model, zero_train_matrix, test_data)
    print(f'Best k Final Test Accuracy: {test_accuracy}')

    print(f'Best Regularization Penalty (lamb): {optimal_lambda}')
    print(f'Best Validation Accuracy: {max_accuracy}')


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################


if __name__ == "__main__":
    main()
