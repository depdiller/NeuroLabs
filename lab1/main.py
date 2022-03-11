import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# x1 = np.array(
#     [[-1, -1, 1, 1, 1],
#      [-1, 1, -1, -1, 1],
#      [1, -1, -1, -1, 1],
#      [1, 1, 1, 1, 1],
#      [1, -1, -1, -1, 1]]
# )
# x2 = np.array(
#     [[-1, 1, 1, 1, -1],
#      [1, -1, -1, -1, 1],
#      [1, -1, -1, -1, -1],
#      [1, -1, -1, -1, 1],
#      [-1, 1, 1, 1, -1]]
# )
# x3 = np.array(
#     [[1, -1, -1, -1, 1],
#      [1, 1, -1, 1, 1],
#      [1, -1, 1, -1, 1],
#      [1, -1, -1, -1, 1],
#      [1, -1, -1, -1, 1]]
# )
x1 = np.array(
    [[-1, -1, 1, 7, 1],
     [-1, 1, -1, -1, 1],
     [1, -1, -1, -1, 1],
     [1, 1, 1, 1, 1],
     [1, -1, -1, -1, 1]]
)
x2 = np.array(
    [[-1, 1, 1, 7, -1],
     [1, -1, -1, -1, 1],
     [1, -1, -1, -1, -1],
     [1, -1, -1, -1, 1],
     [-1, 1, 1, 1, -1]]
)
x3 = np.array(
    [[1, -1, -1, -7, 1],
     [1, 1, -1, 1, 1],
     [1, -1, 1, -1, 1],
     [1, -1, -1, -1, 1],
     [1, -1, -1, -1, 1]]
)

class HammingLayer:
    def __init__(self, neurons_number: int):
        self.neurons_number = neurons_number
        self.neurons = list()
        ones = np.ones(shape=(5, 5))
        for neuron_i in range(neurons_number):
            neuron_i = LinearNeuron(0, ones)

    @staticmethod
    def __dist(weights: np.array, sample: np.array):
        return 2 * (x1.shape[0] * x1.shape[0] - np.multiply(weights, sample).sum())

    def hamming_dist(self, sample: np.array, benchmarks: tuple[np.array]):
        if self.neurons_number != len(benchmarks):
            raise Exception('Wrong size of benchmark array')
        result_dist = list()
        for bench in benchmarks:
            result_dist.append(self.__dist(bench, sample))
        return result_dist

    @staticmethod
    def __hamming_criteria(weights: np.array, sample: np.array):
        return np.multiply(weights, sample).sum() + x1.shape[0] * x1.shape[0]

    def layer_result(self, sample: np.array, benchmarks: tuple[np.array]):
        if self.neurons_number != len(benchmarks):
            raise Exception('Wrong size of benchmark array')
        result_criteria = list()
        for bench in benchmarks:
            result_criteria.append(self.__hamming_criteria(bench, sample))

        return result_criteria

class LinearNeuron:
    def __init__(self, bias: float, weights):
        self.bias = bias
        self.weights = weights

    def run(self, input: tuple[float]):
        if len(input) != len(self.weights):
            raise Exception('Input do not correlate with weights')
        h = sum(tuple(map(lambda el1, el2: el1 * el2, input, self.weights))) + self.bias
        return h if (h > 0) else 0

    def update_weights(self, index: int, new_weights):
        self.weights = new_weights


class StepMaxnetNeuron:
    def __init__(self, bias: float, weights: tuple[float]):
        self.bias = bias
        self.weights = weights

    def run(self, input: tuple[float]):
        if len(input) != len(self.weights):
            raise Exception('Input do not correlate with weights')
        h = sum(tuple(map(lambda el1, el2: el1 * el2, input, self.weights))) + self.bias
        return 1 if h > 0 else 0


class Comparator:
    def __init__(self):
        self.neuron1 = LinearNeuron(0, (1, -1))
        self.neuron2 = LinearNeuron(0, (1, -1))
        self.neuron3 = LinearNeuron(0, (0.5, 0.5, 0.5, 0.5))
        self.neuron4 = StepMaxnetNeuron(0, (1,))
        self.neuron5 = StepMaxnetNeuron(0, (1,))

    def Compare(self, input):
        if len(input) != 2:
            raise Exception('Comparator takes two objects')
        y = list()
        input_reverse = input[::-1]
        v1 = self.neuron1.run(input)
        v2 = self.neuron2.run(input_reverse)

        # TODO rewrite this block
        if v1 == 0 and v2 == 0:
            v1 = 1

        z = self.neuron3.run((v1, v2, input[0], input[1]))
        y.append(self.neuron4.run((v1,)))
        y.append(z)
        y.append(self.neuron5.run((v2,)))
        return y


class OutputNeuron:
    def __init__(self, bias: float, weights: Tuple[float]):
        self.bias = bias
        self.weights = weights

    def run(self, input: Tuple[float]):
        if len(input) != len(self.weights):
            raise Exception('Input do not correlate with weights')
        h = sum(tuple(map(lambda el1, el2: el1 * el2, input, self.weights))) + self.bias
        return 1 if h > 1.5 else 0


class MaxnetFeedForward:
    def __init__(self, classes_number: int):
        self.classes_number = classes_number

    def run(self, benchmarks: Tuple[np.array], sample: np.array):
        hamming_layer = HammingLayer(self.classes_number)
        layer_output = hamming_layer.layer_result(sample, benchmarks)
        print(layer_output)

        # TODO rewrite this block
        result = list()
        if self.classes_number != 3:
            raise Exception('cannot process more than 3 yet')
        # maxnet layers for 3
        c1 = Comparator()
        c2 = Comparator()
        y1 = c1.Compare(layer_output[:2])
        y2 = c2.Compare((y1[1], layer_output[2]))
        neuron11 = OutputNeuron(0, (1, 1))
        neuron12 = OutputNeuron(0, (1, 1))
        neuron13 = StepMaxnetNeuron(0, (1,))
        result.append(neuron11.run((y1[0], y2[0])))
        result.append(neuron12.run((y1[2], y2[0])))
        result.append(neuron13.run((y2[2],)))
        return np.nonzero(result)[0][0] + 1


class MaxnetRecursive:
    def __init__(self, classes_number: int):
        self.classes_number = classes_number
        self.neurons = list()
        self.epsilon = 1.0 / (100*classes_number)

    def set_epsilon(self, eps: float):
        if eps > 1 / self.classes_number:
            raise Exception('epsilon must be >= 1/k')
        self.epsilon = eps

    def epsilon(self):
        return self.epsilon

    def run(self, benchmarks: Tuple[np.array], sample: np.array):
        hamming_layer = HammingLayer(self.classes_number)
        layer_output = hamming_layer.layer_result(sample, benchmarks)

        next_layer = np.empty(shape=(self.classes_number,))
        for i in range(self.classes_number):
            weights = list()
            for j in range(self.classes_number):
                weights.append(1 if (i == j) else -self.epsilon)
            self.neurons.append(LinearNeuron(0, tuple(weights)))

        convergence_rate = 0
        while np.nonzero(layer_output)[0].size != 1:
            for i in range(self.classes_number):
                next_layer[i] = self.neurons[i].run(layer_output)
            layer_output = next_layer
            convergence_rate += 1

        return np.nonzero(layer_output)[0][0] + 1, convergence_rate


# x = np.random.choice(range(-1, 2, 2), size=(5, 5))
x = np.array(
    [[-1, -1, 1, -1, -1],
     [-1, 1, -1, 1, -1],
     [1, -1, -1, -1, 1],
     [1, 1, 1, 1, 1],
     [1, -1, -1, -1, 1]]
)

# x = np.array(
#     [[1, 1, 1, 1, 1],
#     [1, -1, -1, -1, -1],
#     [1, -1, -1, -1, -1],
#     [1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1]]
# )
# x = np.array(
#     [[-1, 1, -1, -1, -1],
#     [-1, -1, -1, -1, -1],
#     [1, 1, 1, -1, 1],
#     [1, -1, 1, 1, 1],
#     [1, -1, 1, -1, 1]]
# )

x_bench = [x1, x2, x3, x]
fig2, axes = plt.subplots(nrows=1, ncols=4)
i = 0
for ax, x_i in zip(axes, x_bench):
    ax.imshow(2 - x_i, cmap='gray')
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5], minor='True')
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5], minor='True')
    ax.yaxis.grid(True, which='minor')
    ax.xaxis.grid(True, which='minor')
    i += 1
    if i < 4:
        ax.set_title("benchmark % i" % i)
    else:
        ax.set_title("sample")
plt.show()

ffn = MaxnetFeedForward(3)
dist = HammingLayer(3)
print('Hamming distance: %s' % dist.hamming_dist(x, (x1, x2, x3)))
print('Feed Forward choice = %d\n\n' % ffn.run((x1, x2, x3), x))

rn = MaxnetRecursive(3)
rn.set_epsilon(1 / 6)
result = rn.run((x1, x2, x3), x)
print('Recursive choice = %d.\nConvergence rate = %d\nepsilon = %f' % (result[0], result[1], rn.epsilon))

# robustness
