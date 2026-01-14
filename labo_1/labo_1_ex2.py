import numpy as np
import matplotlib.pyplot as plt

def predict(a, x):
    y_hat = np.zeros_like(x)
    for i in range(len(a)):
        y_hat += a[i] * x**i
    return y_hat

def loss(y, y_hat):
    return np.sum((y - y_hat)**2)

def derivative_loss(a, x, y):
    y_hat = predict(a, x)
    grad = np.zeros_like(a)

    for i in range(len(a)):
        grad[i] = -2 * np.sum((y - y_hat) * x**i)

    return grad

def main():
    polynome = 5
    a = np.zeros(polynome)

    N = 1000
    mu = 0.001
    loss_matrix = np.zeros(N)

    data = np.array([
        [-0.95,0.02],
        [-0.82,0.03],
        [-0.62,-0.17],
        [-0.43, -0.12],
        [-0.17, -0.37],
        [-0.07, -0.25],
        [0.25,-0.10],
        [0.38,0.14],
        [0.61,0.53],
        [0.79,0.71],
        [1.04,1.53]
    ])

    x = data[:,0]
    y = data[:,1]


    for i in range(N):
        y_hat = predict(a, x)
        loss_matrix[i] = loss(y, y_hat)

        grad = derivative_loss(a, x, y)
        a = a - mu * grad

    print("Coefficients :", a)

    x_plot = np.linspace(-1.25, 1.25, 100)
    y_plot = predict(a, x_plot)

    plt.scatter(x, y, label="Données")
    plt.plot(x_plot, y_plot, color="black", label="Polynôme appris")
    plt.legend()
    plt.show()

    plt.plot(loss_matrix)
    plt.title("Loss")
    plt.show()

if __name__ == "__main__":
    main()
