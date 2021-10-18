import numpy as np
from sklearn.metrics import mean_squared_error
import random

class MySGD():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.w = 0
        self.b = 0
    
    def _get_batch(self, X, y, batch_size):
        '''
        This function is used to pick the batch from the input and output dataset.
        '''
        num_examples = X.shape[0]
        idx = list(range(num_examples))
        random.shuffle(idx)
        for batch_i, i in enumerate(range(0, num_examples, batch_size)):
            j = np.array(idx[i: min(i + batch_size, num_examples)])
            yield batch_i, X[j,:], y[j]

    def _get_gradients(self, X_batch, y_batch):
        '''
        Here, the gradients are calculated
        '''
        N = len(X_batch)
        f = y_batch - (self.w * X_batch + self.b)
        gradient_w = (-2 * X_batch.dot(f.T).sum() / N)
        gradient_b = (-2 * f.sum() / N)

        return gradient_w, gradient_b
    
    def fit(self, X, y, batch_size = 32, epochs = 100, early_stopping = False, thr = 0.001):
        '''
        It's performed the actual training.
        This algorithm fetches the batch and calculates the gradient.
        W and b are updated and the loss is calculated and stored. 
        '''
        history = []
        n_iter = 0

        for e in range(epochs):
            losses=[]
            if e > 1 and early_stopping and np.abs(history[-1]-history[-2]) < thr:
                break
            else:
                for batch_i, X_batch, y_batch in self._get_batch(X, y, batch_size):

                    gradient_w, gradient_b = self._get_gradients(X_batch, y_batch)
                    self.w -= self.learning_rate * gradient_w
                    self.b -= self.learning_rate * gradient_b

                    loss = mean_squared_error(y_batch, (self.w * X_batch + self.b))
                    losses.append(loss)

                if e % 10 == 0:
                    print(f"Epoch: {e}, Loss: {np.mean(losses)})")

                history.append(np.mean(losses))
            n_iter += 1
                                      
        return history, n_iter
        
                
    def predict(self, X):
        '''
        This method predict values for the input.
        '''
        return self.w * X + self.b

