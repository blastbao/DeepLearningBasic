class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        inital perceptron
        :param input_num: number of input
        :param activator: type:double -> double
        '''
        self.activator = activator
        # weights are 0
        self.weights = [0.0 for _ in range(input_num)]
        # bias is 0
        self.bias = 0


    def __str__(self):
        '''
        print weights and bias
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)


    def predict(self, input_vec):
        '''
        input vector, output result
        '''
        # zip input_vex[x1, x2, x3 ...] and weights[w1, w2, w3 ...]
        # create list[(x1,w1), (x2,w2), (x3,w3) ...]
        #
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x*w,
                       zip(input_vec, self.weights))
                   , 0.0) + self.bias)


    def train(self, input_vecs, labels, iteration, rate):
        '''
        :param input_vecs:
        :param labels:
        :param iteration:
        :param rate:
        :return:
        '''
        for i in range(iteration):
            self._one_interation(input_vecs, labels, rate)


    def _one_interation(self, input_vecs, labels, rate):
        '''
        one interation, train all the data
        :param input_vecs:
        :param labels:
        :param rate:
        :return:
        '''
        # zip input and output, create list[(input_vex, label), ...]
        # (input_vec, label) is the train sample
        samples = zip(input_vecs, labels)
        # update the weight for every sample by perceptron rules
        for (input_vec, label) in samples:
            # compute the output in current weight
            output = self.predict(input_vec)
            # update weight
            self._update_weights(input_vec, output, label, rate)


    def _update_weights(self, input_vec, output, label, rate):
        '''
        update the weight for every sample by perceptron rules
        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        '''
        # zip input_vec[x1, x2, x3, ...] and weights[w1, w2, w3, ...]
        # create list[(x1, w1), (x2, w2), (x3, w3), ...]
        # update the weight for every sample by perceptron rules
        delta = label - output
        self.weights = map(
            lambda (x,w): w + rate * delta * x,
            zip(input_vec, self.weights))
        # update bias
        self.bias += rate * delta


def f(x):
    '''
    define activator
    :return:
    '''
    return 1 if x > 0 else 0


def get_train_dataset():
    '''
    create train data based on AND table
    :return:
    '''
    # create train data
    # input vector list
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # expect output
    labels = [1, 0, 0, 1]
    return input_vecs, labels


def train_and_perceptron():
    '''
    train perceptron
    :return:
    '''
    # create perceptron, number of input is 2, activator is f
    p = Perceptron(2, f)
    # training, iteration = 10, rate = 0.1
    input_vecs, labels = get_train_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # return trained perceptron
    return p


if __name__ == '__main__':
    # train AND perceptron
    and_perceptron = train_and_perceptron()
    # print weights
    print and_perceptron
    # test
    print '1 and 1 = %d' % and_perceptron.predict([1, 1])
    print '0 and 0 = %d' % and_perceptron.predict([0, 0])
    print '1 and 0 = %d' % and_perceptron.predict([1, 0])
    print '0 and 1 = %d' % and_perceptron.predict([0, 1])



