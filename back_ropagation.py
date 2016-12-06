# Node class
class Node(object)
    def __int__(self, layer_index, node_index):
        '''
        create node boject
        :param layer_index:
        :param node_index:
        :return:
        '''
        self.layer_index = layer_index
        self.node_index = layer_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        set node output
        :param output:
        :return:
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        apend downstream
        :param conn:
        :return:
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        apend upstream
        :param conn:
        :return:
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        compute node output
        :return:
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        compute hidden layer delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        compute output layer delta
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        print node info
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        create const node
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
    def append_downstream_connection(self, conn):
        '''
        apend downstream
        '''
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        '''
        compute hidden layer delta
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta
    def __str__(self):
        '''
        print node info
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str



class Layer(object):
    def __init__(self, layer_index, node_count):
        '''
        init layer
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))
    def set_output(self, data):
        '''
        set layer output
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])
    def calc_output(self):
        '''
        compute output vector
        '''
        for node in self.nodes[:-1]:
            node.calc_output()
    def dump(self):
        '''
        print layer info
        '''
        for node in self.nodes:
            print node


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        init connection
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0
    def calc_gradient(self):
        '''
        compute gradient
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output
    def get_gradient(self):
         '''
         get current gradient
         '''
        return self.gradient
    
    def update_weight(self, rate):
        '''
        update weight
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient
    def __str__(self):
        '''
        print connect info
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

class Connections(object):
    def __init__(self):
        self.connections = []
    def add_connection(self, connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print conn


class Network(object):
    def __init__(self, layers):
        '''
        init network
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0;
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    def train(self, labels, data_set, rate, iteration):
        '''
        train network
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
    def train_one_sample(self, label, sample, rate):
        '''
        train one sample
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)
    def calc_delta(self, label):
        '''
        calc delta
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
    def update_weight(self, rate):
        '''
        update weight
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)
    def calc_gradient(self):
        '''
        calc_gradient
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
    def get_gradient(self, label, sample):
        '''
        get_gradient
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
    def predict(self, sample):
        '''
        predict output
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])
    def dump(self):
        '''
        print network info
        '''
        for layer in self.layers:
            layer.dump()


def gradient_check(network, sample_feature, sample_label):
    '''
    gradient_check
    '''
    # calc network error
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b,
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))
    # get_gradient
    network.get_gradient(sample_feature, sample_label)
    # gradient_check
    for conn in network.connections.connections:
        # get_gradient
        actual_gradient = conn.get_gradient()
        # network_error
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # network_error
        conn.weight -= 2 * epsilon # notice
        error2 = network_error(network.predict(sample_feature), sample_label)
        # expected_gradient
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # print
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient)



class Loader(object):
    def __init__(self, path, count):
        '''
        init
        '''
        self.path = path
        self.count = count
    def get_file_content(self):
        '''
        read file
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content
    def to_int(self, byte):
        '''
        unsigned byte to int
        '''
        return struct.unpack('B', byte)[0]
#

class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        get_picture
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start + i * 28 + j]))
        return picture
    def get_one_sample(self, picture):
        '''
        get_one_sample
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    def load(self):
        '''
        load
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set

class LabelLoader(Loader):
    def load(self):
        '''
        load
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels
    def norm(self, label):
        '''
        norm
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.95)
            else:
                label_vec.append(0.05)
        return label_vec


def get_training_data_set():
    '''
    get_training_data_set
    '''
    image_loader = ImageLoader('train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()

def get_test_data_set():
    '''
    get_test_data_set
    '''
    image_loader = ImageLoader('t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

def get_result(vec):
    # vec
    max_value_index = 0
    max_value = 0
    for i in len(vec):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    correct = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label == predict:
            correct += 1
    return float(correct) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10])
    while True:
        epoch += 10
        network.train(train_data_set, train_labels, 0.1, 10)
        error_ratio = evaluate(network, test_data_set, test_labels)
        print 'after epoch %d, error ratio is %f' % (epoch, error_ratio)
        if error_ratio > last_error_ratio:
            break
        else:
            last_error_ratio = error_ratio
