import numpy as np
import matplotlib.pyplot as plt

def loss(A, B) -> float:
    BA = B @ A
    I = np.eye(6)
    return np.sum((BA-I)**2)

def derivative_loss(A, B):
    BA = B @ A
    I = np.eye(6)
    return 2 * (BA - I) @ A.T

def main():
    A = np.array([[3, 4, 1], [5, 2, 3], [6, 2, 2]])
    A_2 = np.array ([[3,4,1,2,1,5],
                     [5,2,3,2,2,1],
                     [6,2,2,6,4,5],
                     [1,2,1,3,1,2],
                     [1,5,2,3,3,3],
                     [1,2,2,4,2,1]])
    B_random = np.random.rand(3, 3)
    B_random_2 = np.random.rand(6, 6)
    N = 1000
    mu = 0.001

    loss_matrix = np.zeros(N)


    for n in range(N):
        loss_matrix[n] = loss(A_2, B_random_2)
        derive_cout = derivative_loss(A_2, B_random_2)
        B_random_2 -= mu * derive_cout

    np.set_printoptions(suppress=True, precision=4)
    print (B_random_2)
    plt.figure()
    plt.plot(loss_matrix[2:])
    plt.show()

if __name__ == "__main__":
    main()