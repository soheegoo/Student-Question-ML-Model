import sys
sys.path.append("../")
from part_a import item_response
from torch import sigmoid
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils import *


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

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (mle_train_matrix, question_train_matrix, valid_data, test_data)
        WHERE:
        mle_train_matrix: 2D sparse matrix where missing entries are
        filled with the mean of the row
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

    # fill in the missing entries with mean value
    mle_train_matrix = question_train_matrix.copy()
    num_question = question_train_matrix.shape[0]
    for i in range(num_question):
        row_mean = np.nanmean(question_train_matrix[i])
        nan_mask = np.isnan(question_train_matrix[i])
        mle_train_matrix[i, nan_mask] = row_mean

    mle_train_matrix = torch.FloatTensor(mle_train_matrix)

    question_train_matrix = torch.FloatTensor(question_train_matrix)

    return mle_train_matrix, question_train_matrix, valid_data, test_data

class Dataset(TensorDataset):
    def __init__(self, mle_train_matrix, beta_vector, meta_data) -> None:
        """
        :param mle_train_matrix: matrix with nan replaced by mean
        :param beta_vector: vector
        :param meta_data: 2D FloatTensor
        """
        super().__init__()
        self.mle_train_matrix = mle_train_matrix
        self.beta_vector = beta_vector
        self.meta_data = meta_data

    def __len__(self):
        return self.mle_train_matrix.shape[0]

    def __getitem__(self, index):
        """
        :param index: question index
        """
        return {'question_id': index,
                'question_vector': self.mle_train_matrix[index],
                'beta': torch.tensor([self.beta_vector[index]], dtype=torch.float32) if
                self.beta_vector is not None else torch.nan,
                'meta_vector': self.meta_data[index]
                if self.meta_data is not None else torch.nan
                }

'''
This is a question based autoencoder instead of a student based autoencoder which takes
in question vectors as inputs.
We can add 2 more hidden layers to make it a 5 layer neural net architecture.
We can optionally pass in the beta value which is added to the latent vector. 
We can optionally pass in the meta data values which is added to the latent vector. 
'''
class AutoEncoder(nn.Module):
    def __init__(self, num_students, num_subjects, k=100, j=10, beta_latent_dim=1,
                 subject_latent_dim=5):
        """ Initialize a class AutoEncoder.

        :param num_students: int
        :param num_subjects: int
        :param k: int
        :param j: int
        :param beta_latent_dim: int
        :param subject_latent_dim: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        # Adding two more hidden layers to help with underfitting
        self.g1 = nn.Linear(num_students, k)

        self.g2 = nn.Linear(k, j)
        self.h2 = nn.Linear(j + beta_latent_dim +
                            subject_latent_dim, k)

        self.h1 = nn.Linear(k, num_students)

        self.subject_enc_linear = nn.Linear(num_subjects, subject_latent_dim)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2 + ||W^3||^3 + ||W^4||^4

        :return: float
        """
        g2_w_norm, h2_w_norm = 0, 0
        g2_w_norm = torch.norm(self.g2.weight, 2) ** 2
        h2_w_norm = torch.norm(self.h2.weight, 2) ** 2
        g1_w_norm = torch.norm(self.g1.weight, 2) ** 2
        h1_w_norm = torch.norm(self.h1.weight, 2) ** 2

        return g1_w_norm + g2_w_norm + h1_w_norm + h2_w_norm

    def forward(self, inputs, beta=None, meta_data=None):
        """ Return a forward pass given inputs.

        :param inputs: question vector.
        :param beta: beta value
        :param meta_data: meta data vector
        :return: question vector.
        """
        question_raw_latent = F.sigmoid(self.g1(inputs))
        question_latent = F.sigmoid(self.g2(question_raw_latent))

        if beta is not None:
            beta_tensor = torch.tensor([[beta]], dtype=torch.float32)
            combined_latent = torch.cat((question_latent, beta_tensor), dim=-1)

        if meta_data is not None:
            subject_latent = torch.sigmoid(self.subject_enc_linear(meta_data))
            subject_latent = Variable(subject_latent).unsqueeze(0)
            combined_latent = torch.cat(
                (combined_latent, subject_latent), dim=-1)
        else:
            combined_latent = question_latent

        # decode
        combined_latent = F.sigmoid(self.h2(combined_latent))
        decoded = F.sigmoid(self.h1(combined_latent))

        return decoded


def train(model, lr, lamb, question_train_data, mle_train_data, valid_data, num_epoch, batch_size,
          betas=None, metas=None):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param question_train_data: 2D FloatTensor
    :param mle_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param batch_size: int
    :param betas: vector
    :param metas: 2D FloatTensor
    :return: (model, train_losses, val_accuracies)
        WHERE:
        model: trained autoencoder
        train_losses: list
        val_accuracies: list
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Build dataset object
    dataset = Dataset(
        mle_train_matrix=mle_train_data,
        beta_vector=betas,
        meta_data=metas
    )

    # Define dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for data_points in data_loader:
            question_id_batch = data_points['question_id']
            question_vectors_batch = data_points['question_vector']
            beta_batch = data_points['beta']
            meta_batch = data_points['meta_vector']

            optimizer.zero_grad()

            loss = 0

            if betas is not None or metas is not None:

                for i in range(len(question_id_batch)):
                    single_input = question_vectors_batch[i]

                    single_beta = beta_batch[i] if beta_batch is not None else None

                    single_meta = meta_batch[i] if meta_batch is not None else None

                    inputs = Variable(single_input).unsqueeze(0)
                    target = inputs.clone()
                    question_id = question_id_batch[i]
                    nan_mask = np.isnan(
                        question_train_data[question_id].unsqueeze(0).numpy())
                    output = model(inputs, single_beta, single_meta)
                    target[0][nan_mask] = output[0][nan_mask]
                    loss += torch.sum((output - target) ** 2.) + (lamb / 2) * (
                        model.get_weight_norm())

            else:
                inputs = Variable(question_vectors_batch).unsqueeze(0)
                target = inputs.clone()

                output = model(inputs)

                # Mask the target to replace missing values with the corresponding values
                # from the output
                nan_mask = np.isnan(
                    question_train_data[question_id_batch].unsqueeze(0).numpy())
                target[0][nan_mask] = output[0][nan_mask]

                loss = torch.sum((output - target) ** 2.) + \
                    (lamb / 2) * (model.get_weight_norm())

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, mle_train_data, valid_data, betas, metas)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_losses.append(train_loss)
        val_accuracies.append(valid_acc)

    return model, train_losses, val_accuracies


def evaluate(model, train_data, valid_data, betas, metas):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param betas: vector
    :param metas: 2D FloatTensor
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[q].unsqueeze(0))
        beta = betas[q] if betas is not None else None
        meta = metas[q] if metas is not None else None

        output = model(inputs, beta, meta)

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    mle_train_matrix, question_train_matrix, valid_data, test_data = load_data()
    train_data = load_train_csv("../data")

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

    # Pre-train IRT model
    _, betas, _, _ = item_response.irt(
        train_data=train_data, val_data=valid_data, lr=0.004, iterations=160)

    # Training the model with the best hyperparameters
    k_star = 100
    lr = 0.01
    num_epoch = 10
    lamb = 0.001
    js = [5, 10, 15, 20]
    batch_sizes = [5, 10, 30]
    meta_latent_dim_list = [5]

    print(f"Training model with K = {k_star}, Learning Rate = {lr}"
    f", Epochs = {num_epoch}, Lam = {lamb}")

    max_accuracy = 0
    optimal_model = None
    optimal_train_losses = []
    optimal_val_accuracies = []
    optimal_j = 0
    optimal_batch = 0

    for batch_size in batch_sizes:
        for meta_latent_dim in meta_latent_dim_list:
            for j in js:
                beta_latent_dim = 1 if betas is not None else 0

                model = AutoEncoder(num_students=question_train_matrix.shape[1],
                                    num_subjects=num_subject, k=k_star,
                                    j=j, beta_latent_dim=beta_latent_dim,
                                    subject_latent_dim=meta_latent_dim)


                model, train_loss, val_acc = train(model, lr, lamb, question_train_matrix,
                                                   mle_train_matrix,
                                                   valid_data, num_epoch, batch_size, betas, meta_data)

                valid_accuracy = val_acc[-1]

                if valid_accuracy > max_accuracy:
                    max_accuracy = valid_accuracy
                    optimal_train_losses = train_loss
                    optimal_val_accuracies = val_acc
                    optimal_model = model
                    optimal_j = j
                    optimal_batch = batch_size

                print(f"Training model with j = {j}, Batch Size = {batch_size}")
                print(f'Validation Accuracy: {valid_accuracy}')
                print(f'===============================================')
    
    test_accuracy = evaluate(
                    optimal_model, mle_train_matrix, test_data, betas, metas)
    print(f'Test Accuracy: {test_accuracy}')


if __name__ == "__main__":
    main()
