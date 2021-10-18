import numpy as np
from sklearn.metrics import mean_squared_error
import random

class MyAdam():
    def __init__(self, learning_rate, momentum_decay = [0.9, 0.999],  epsilon = 10 ** -8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.momentum_decay_1 = momentum_decay[0]
        self.momentum_decay_2 = momentum_decay[1]
        self.w = 0
        self.b = 0
        self.scale_w = 0
        self.scale_b = 0
        self.momentum_vector_w = 0 
        self.momentum_vector_b = 0

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
        N=len(X_batch)
        f = y_batch - (self.w * X_batch + self.b) 
        gradient_w =(-2 * X_batch.dot(f.T).sum() / N) 
        gradient_b = (-2 * f.sum() / N)
        
        return gradient_w, gradient_b
    
    def _get_momentum_vector(self, X_batch, y_batch, gradient_w, gradient_b):
        '''
        It updates biased first moment estimate 
        '''
        self.momentum_vector_w = (self.momentum_decay_1 * self.momentum_vector_w) + ((1 - self.momentum_decay_1) * gradient_w)
        self.momentum_vector_b = (self.momentum_decay_1 * self.momentum_vector_b) + ((1 - self.momentum_decay_1) * gradient_b)
        
   
    def _get_scale(self, X_batch, y_batch, gradient_w, gradient_b):
        '''
        It updates biased second moment estimate 
        '''
        self.scale_w = (self.momentum_decay_2 * self.scale_w) + ((1 - self.momentum_decay_2) * np.multiply(gradient_w, gradient_w))
        self.scale_b = (self.momentum_decay_2 * self.scale_b) + ((1 - self.momentum_decay_2) * np.multiply(gradient_b, gradient_b))
        
    
        
    def fit(self, X, y, batch_size = 32, epochs = 100, early_stopping = False, thr = 0.001):
        '''
        Here it's performed the actual training.
        For each epoch, this algorithm fetches the batch, calculates the gradients, the momentum vectors and the scale vector.
        W and b are updated using these vectors. 
        The loss is calculated and stored. 
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

                    self._get_momentum_vector(X_batch, y_batch,gradient_w, gradient_b)#mt
                    self._get_scale(X_batch, y_batch, gradient_w, gradient_b)#vt

                    #Compute bias-corrected first moment estimate
                    mdw_corr=self.momentum_vector_w/(1-np.power(self.momentum_decay_1, e+1))#^mt
                    mdb_corr=self.momentum_vector_b/(1-np.power(self.momentum_decay_1, e+1))#^mt
                    #Compute bias-corrected second moment estimate 
                    vdw_corr=self.scale_w/(1-np.power(self.momentum_decay_2, e+1))#^vt
                    vdb_corr=self.scale_b/(1-np.power(self.momentum_decay_2, e+1))#^vt

                    #Update parameters
                    divider_w = np.sqrt(vdw_corr) + self.epsilon
                    divider_b = np.sqrt(vdb_corr) + self.epsilon
                    self.w -= self.learning_rate * mdw_corr / divider_w
                    self.b -= self.learning_rate * mdb_corr/ divider_b

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