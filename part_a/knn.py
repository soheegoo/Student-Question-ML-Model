from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from starter_code.utils import *



def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    for i in range(2):
        k_array = np.linspace(1, 26, 6).astype(int)
        if i == 0:
            user_k_acc = []
            print("Method: Knn Impute by User")
            for k in k_array:
                acc = knn_impute_by_user(matrix=sparse_matrix, valid_data=val_data, k=k)
                user_k_acc.append(acc)
            user_k_acc_arr = np.array(user_k_acc)
            k_index = np.argmax(user_k_acc_arr)
            final_acc = knn_impute_by_user(matrix=sparse_matrix, valid_data=test_data, k=k_array[
                k_index])

            print("Best K value is " + str(k_array[k_index]))
            print("Validation Accuracy with best k is " + str(user_k_acc_arr[k_index]))
            print("Test Accuracy is " + str(final_acc))

            plt.figure(figsize=(8, 4))
            plt.plot(k_array, user_k_acc_arr)
            plt.scatter(k_array, user_k_acc_arr)
            plt.xlabel('K Value')
            plt.ylabel('Validation accuracy')
            plt.title(f'KNN accuracy: Similarity by User')

            plt.show()

        else:
            item_k_acc = []
            print("Method: Knn Impute by Item")
            # iterate through all ks
            for k in k_array:
                acc = knn_impute_by_item(matrix=sparse_matrix, valid_data=val_data, k=k)
                item_k_acc.append(acc)
            item_k_acc_arr = np.array(item_k_acc)
            k_index = np.argmax(item_k_acc_arr)
            final_acc = knn_impute_by_item(matrix=sparse_matrix, valid_data=test_data, k=k_array[
                k_index])

            print("Best K value is " + str(k_array[k_index]))
            print("Validation Accuracy with best k is " + str(item_k_acc_arr[k_index]))
            print("Test Accuracy is " + str(final_acc))

            plt.figure(figsize=(8, 4))
            plt.plot(k_array, item_k_acc_arr)
            plt.scatter(k_array, item_k_acc_arr)
            plt.xlabel('K Value')
            plt.ylabel('Validation accuracy')
            plt.title(f'KNN accuracy: Similarity by Item')

            plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
