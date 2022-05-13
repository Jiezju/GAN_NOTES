import numpy as np


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
