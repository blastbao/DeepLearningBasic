from perceptron import  Perceptron

# activation function
f = lambda x:x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''init linear unit, set input'''
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    # input list
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # except output
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = LinearUnit(1)
    # train linear unit
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    # print weights
    print linear_unit
    # test
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])