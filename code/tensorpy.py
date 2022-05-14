import numpy as np
from queue import Queue

class Graph:
    def __init__(self):
        self.operations = []  # 操作节点
        self.placeholders = []  # 占位符节点
        self.variables = []  # 变量节点

    def as_default(self):
        global _default_graph
        _default_graph = self


class Operation:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes  # 输入该操作的节点
        # 消费者列表 表示 哪些变量 使用这个操作
        self.consumers = []

        for input_node in input_nodes:
            input_node.consumers.append(self)

        # 加入默认计算图
        _default_graph.operations.append(self)

    def compute(self):
        pass


class placeholder:
    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)


class Variables:
    def __init__(self, initil_value=None):
        self.value = initil_value
        self.consumers = []
        _default_graph.variables.append(self)


class matmul(Operation):
    def __init__(self, x, y):
        super(matmul, self).__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value.dot(y_value)


class add(Operation):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value + y_value

class sigmoid(Operation):
    def __init__(self, x):
        super(sigmoid, self).__init__([x])

    def compute(self, x_value):
        return 1 / (1 + np.exp(-x_value))

class softmax(Operation):
    def __init__(self, x):
        super(softmax, self).__init__([x])

    def compute(self, x_value):
        return np.exp(x_value) / np.sum(np.exp(x_value), axis=1)[:, None]

class negative(Operation):
    def __init__(self, x):
        super(negative, self).__init__([x])

    def compute(self, x_value):
        return -x_value

class reduce_sum(Operation):
    def __init__(self, A, axis=None):
        super(reduce_sum, self).__init__([A])
        self.axis = axis

    def compute(self, A_value):
        return np.sum(A_value, self.axis)

class multiply(Operation):
    def __init__(self, x, y):
        super(multiply, self).__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value * y_value

class log(Operation):
    def __init__(self, x):
        super(log, self).__init__([x])

    def compute(self, x_value):
        return np.log(x_value)


# 梯度计算
_gradient_registry = {}
class ResiterGradient:
    def __init__(self, op_type):
        self.op_type = eval(op_type)

    def __call__(self, f):
        _gradient_registry[self.op_type] = f
        return f

# negative_gradients
@ResiterGradient('negtive')
def _negative_gradient(op, grad):
    return -grad

def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1

    # 广度优先搜索
    visited = set()
    queue = Queue()

    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        if node != loss:
            grad_table[loss] = 0

            # 计算所有梯度
            for consumer in node.consumers:
                loss_grad_consumer_output = grad_table[consumer]
                bp_func = _gradient_registry[consumer.__class__]
                loss_grad_consumer_inputs = bp_func(consumer, loss_grad_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += loss_grad_consumer_inputs
                else:
                    node_index_consumer_inputs = consumer.input_nodes.index(node)
                    loss_grad = loss_grad_consumer_inputs[node_index_consumer_inputs]
                    grad_table[node] += loss_grad

        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


# 优化器
class MinimizationOperation(Operation):
    def __init__(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate

    def compute(self):
        # 计算梯度， grad_table 是 dict ： key: node value: grad
        grad_table = compute_gradients(loss)

        # 遍历所有节点
        for node in grad_table:
            if isinstance(node, Variables):
                grad = grad_table[node]

                node.value -= self.learning_rate * grad

class GradientDescentOptimizer:
    def __init__(self, learning_rate, loss):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        return MinimizationOperation(loss, self.learning_rate)


def traverse_postorder(operation):
    nodes_postorder = []

    def recursion(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                # 递归调用，获取所有节点
                recursion(input_node)
        nodes_postorder.append(node)

    recursion(operation)

    return nodes_postorder


class Session:
    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if isinstance(node, placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variables):
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if isinstance(node.output, list):
                node.output = np.array(node.output)

        return operation.output


if __name__ == "__main__":
    g = Graph()
    g.as_default()
    a = Variables([[2, 1], [-1, -2]])
    b = Variables([1, 1])
    c = placeholder()

    # 这里只是初始化计算图
    y = matmul(a, b)
    z = add(y, c)
    print(z)

    # 计算
    session = Session()
    output = session.run(z, {c: [3, 3]})
    print(output)
