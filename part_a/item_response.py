import sys
sys.path.append("../")
from utils import *
import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    length = len(data["user_id"])
    for index in range(length):
        question_id = data["question_id"][index]
        user_id = data["user_id"][index]
        difference = theta[user_id] - beta[question_id]
        c = data["is_correct"][index]
        log_lklihood += c*difference - np.log(1.0+np.exp(difference))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    N = len(theta)
    M = len(beta)

    length = len(data["user_id"])

    gradient_theta = np.zeros(N)
    gradient_beta = np.zeros(M)

    for index in range(length):
        i = data["user_id"][index]
        j = data["question_id"][index]
        cij = data["is_correct"][index]
        theta_i = theta[i]
        beta_j = beta[j]
        gradient_theta[i] = gradient_theta[i] + (cij - sigmoid(theta_i-beta_j))
        gradient_beta[j] = gradient_beta[j] + (-1*cij+sigmoid(theta_i-beta_j))

    theta -= lr * (-1*gradient_theta)
    beta -= lr * (-1*gradient_beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_log_likelihood, train_log_likelihood = [], []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_log_likelihood.append(train_neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likelihood.append(val_neg_lld)
        
        theta, beta = update_theta_beta(train_data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.004
    iterations = 160
    theta, beta, val_log_likelihood, train_log_likelihood = irt(
        train_data, val_data, lr, iterations)

    # plotting the training and validation log likelihoods as function of iterations
    x = [i for i in range(iterations)]
    plt.plot(x, train_log_likelihood, label="train")
    plt.plot(x, val_log_likelihood, label="valid")
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Iterations")
    plt.title("Neg Log Likelihood for Train and Validation Data")
    plt.legend()
    plt.show()

    # reporting the final validation and test accuracies
    val_score = evaluate(data=val_data, theta=theta, beta=beta)
    test_score = evaluate(data=test_data, theta=theta, beta=beta)

    print("Validation Accuracy: ", val_score)
    print("Test Accuracy: ", test_score)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # select 3 questions and plot three curves showing p(c_ij = 1) as a
    # function of theta given j
    population = np.arange(1774)
    q_list = random.sample(population.tolist(), 3)
    theta_range = np.linspace(-5.0, 5.0, 100)

    for selected in q_list:
        plt.plot(theta_range, sigmoid(theta_range -
                 beta[selected]), label=f"Question {selected}")

    plt.ylabel("p(c_ij) = 1")
    plt.xlabel("Theta")
    plt.title("p(c_ij) as a function of theta")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
