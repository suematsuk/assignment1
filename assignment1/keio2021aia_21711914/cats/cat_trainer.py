from . import *

class Cat_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.model.train_mean = self.dataset.train_mean
        self.model.train_sd = self.dataset.train_sd
        self.loss = lrloss

    def accuracy(self, data):

        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])

    def train(self, lr, ne):
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs = []
        accuracies = []

        for epoch in range(1, ne+1):
              J = 0
            for x, y in self.dataset:
                yhat = self.model(x)
                self.model.w += lr*(y - yhat)*x
                self.model.b += lr*(y - yhat)

            accuracy = self.accuracy(self.dataset)
            if epoch%10 == 0:
                print('--> epoch=%d, accuracy=%.3f' % (epoch, accuracy))
            costs.append(J)
            accuracies.append(accuracy)

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))
        
        costs = list(map(lambda t: np.mean(t), [np.array(costs)[i-10:i+11] for i in range(1, len(costs)-10)]))
        accuracies = list(map(lambda t: np.mean(t), [np.array(accuracies)[i-10:i+11] for i in range(1, len(accuracies)-10)]))
        
        return (costs, accuracies)
