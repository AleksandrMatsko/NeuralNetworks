import math


def me(x: float) -> float:
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def tanh(x: float) -> float:
    return math.tanh(x)


def relu(x: float) -> float:
    if x > 0:
        return x
    return 0


def derivative(f, x: float, h: float):
    """
        Approximates the first derivative with O(h^2)
    """
    return (f(x + h) - f(x - h)) / (2 * h)
