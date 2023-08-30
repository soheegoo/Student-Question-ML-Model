# TODO: complete this file.
import sys
sys.path.append("../")
from matplotlib import pyplot as plt
from item_response import *
from neural_network import *
from utils import *
from sklearn.impute import KNNImputer
import torch
from torch.autograd import Variable
import torch.utils.data

# hyperparameters
NEURAL_NET = {
    'k': 10,
    'lr': 0.1,
    'lambda': 0.001,
    'epoch': 10
}
ITEM_RT = {
    'lr': 0.004,
    'iter': 160
}

def bootstrap_method(data):
    # initialize bootstrapped data
    bootstrapped_data = {}

    # turn values from original data into numpy arrays
    questions = np.array(data['question_id'])
    is_correct = np.array(data['is_correct'])
    users = np.array(data['user_id'])

    # n = num of samples in data
    n = len(questions)

    # uniformly select indices (with replacement)
    i_array = np.random.choice(range(0, n), n, True)
    bootstrapped_data['question_id'] = questions[i_array]
    bootstrapped_data['is_correct'] = is_correct[i_array]
    bootstrapped_data['user_id'] = users[i_array]
    return bootstrapped_data


def convert_to_matrix(data, user_ids, question_ids):
    # initialize empty matrix and convert train data into matrix
    matrix = np.zeros((np.max(user_ids) + 1, np.max(question_ids) + 1))
    matrix[:] = np.nan

    # iterate through training data and populate matrix
    for i in range(0, len(data['user_id'])):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]
        matrix[user_id, question_id] = is_correct
    return matrix

def nn_pred(train_matrix, test_data):
    # use torch and get train matrix
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    # train nn
    nn_model = AutoEncoder(train_matrix.shape[1], NEURAL_NET['k'])
    nn_model.train()
    nn_model, train_loss, val_acc = train(nn_model, NEURAL_NET['lr'], NEURAL_NET['lambda'], train_matrix, zero_train_matrix,
          test_data, NEURAL_NET['epoch'])

    # making predictions
    nn_model.eval()
    predictions = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        output = nn_model(inputs)
        guess = output[0][test_data["question_id"][i]].item() >= 0.5
        predictions.append(guess)

    return np.array(predictions).astype('float32')



def irt_pred(train_data, test_data):
    # train irt model
    theta, beta, d1, d2 = \
        irt(train_data, test_data, ITEM_RT['lr'], ITEM_RT['iter'])

    # generate predictions
    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        predictions.append(p_a >= 0.5)
    return np.array(predictions).astype('float32')


def knn(train_matrix, test_data):
    # train knn model
    nbrs = KNNImputer(n_neighbors=11)
    mat = nbrs.fit_transform(train_matrix)

    # generate predictions
    predictions = []
    for i in range(len(test_data['user_id'])):
        user_ids = test_data['user_id'][i]
        q_ids = test_data['question_id'][i]
        predictions.append(mat[user_ids, q_ids])

    return np.array(predictions).astype('float32')


def evaluate(predictions, test_data):
    # evaluate model and find its acc
    prediction = (np.array(predictions >= 0.5))
    targets = np.array(test_data['is_correct'])
    return np.sum(prediction == targets) / len(predictions)


def main():
    validation_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_data = load_train_csv("../data")

    user_ids = train_data['user_id']
    question_ids = train_data['question_id']

    # convert training data to matrix and use bootstrap
    matrix_1 = convert_to_matrix(bootstrap_method(train_data), user_ids, question_ids)
    irt_train_data = bootstrap_method(train_data)
    matrix_3 = convert_to_matrix(bootstrap_method(train_data), user_ids, question_ids)

    # validation data

    val_nn_result = nn_pred(matrix_1, validation_data)
    val_irt_result = irt_pred(irt_train_data, validation_data)
    val_knn_result = knn(matrix_3, validation_data)

    val_average = (val_nn_result + val_irt_result + val_knn_result) / 3
    bagged_val_accuracy = evaluate(val_average, validation_data)

    # test data 

    test_nn_result = nn_pred(matrix_1, test_data)
    test_irt_result = irt_pred(irt_train_data, test_data)
    test_knn_result = knn(matrix_3, test_data)

    # get average over 3
    test_average = (test_nn_result + test_irt_result + test_knn_result) / 3
    bagged_test_accuracy = evaluate(test_average, test_data)

    # display final test and val accuracy
    print("test accuracy: " + str(bagged_test_accuracy))
    print("val accuracy: " + str(bagged_val_accuracy))


if __name__ == '__main__':
    main()
