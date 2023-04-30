import math


def me(x: float) -> float:
    return x


def derivative_me(x: float) -> float:
    return 1


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x: float) -> float:
    tmp = sigmoid(x)
    return math.exp(-x) * tmp * tmp


def tanh(x: float) -> float:
    return math.tanh(x)


def derivative_tanh(x: float) -> float:
    tmp = math.cosh(x)
    return 1 / (tmp * tmp)


def relu(x: float) -> float:
    if x > 0:
        return x
    return 0


def leaky_relu(x: float) -> float:
    if x > 0:
        return x
    return 0.01 * x


def derivative(f, x: float, h: float):
    """
        Approximates the first derivative with O(h^2) if there is no ready derivative
    """
    d_dx = DERIVATIVE_DICT.get(f)
    if d_dx is not None:
        return d_dx(x)
    return (f(x + h) - f(x - h)) / (2 * h)


FUNC_DICT = {
    "ReLu": relu,
    "tanh": tanh,
    "me": me,
    "sigmoid": sigmoid,
    "LeakyReLu": leaky_relu
}

DERIVATIVE_DICT = {
    me: derivative_me,
    sigmoid: derivative_sigmoid,
    tanh: derivative_tanh,
}
