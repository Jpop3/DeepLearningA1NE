import numpy as np

class MiniBatch(object):

    def __init__(self, train_x, train_y, batch_size):
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size
        self.no_of_batches = int(self.train_x.shape[0]/self.batch_size) #has to be a nice number
        self.batches_x = np.zeros((int(self.no_of_batches), int(self.batch_size), 128))
        self.batches_y = np.zeros((int(self.no_of_batches), int(self.batch_size), 10))
        self.create_batches()

    def shuffle(self):
        p = np.random.permutation(len(self.train_x))
        self.train_x = self.train_x[p]
        self.train_y = self.train_y[p]
    
    def create_batches(self):
        self.shuffle() #This fixed it. I believe it was learning the ordering of the mini batches and then made bad edge weights
        i = 0
        while (i + 1) * self.batch_size < self.train_x.shape[0]: #might have to adjust by 1?
            self.batches_x[i] = self.train_x[i:i+self.batch_size]
            self.batches_y[i] = self.train_y[i:i+self.batch_size]
            i += 1


