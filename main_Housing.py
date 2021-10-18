from sgd import *
from adam import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

'''
Data Extraction
Columns Details:
-CRIM - per capita crime rate by town
-ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
-INDUS - proportion of non-retail business acres per town.
-CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
-NOX - nitric oxides concentration (parts per 10 million)
-RM - average number of rooms per dwelling
-AGE - proportion of owner-occupied units built prior to 1940
-DIS - weighted distances to five Boston employment centres
-RAD - index of accessibility to radial highways
-TAX - full-value property-tax rate per $10,000
-PTRATIO - pupil-teacher ratio by town
-B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
-LSTAT - % lower status of the population
-MEDV - Median value of owner-occupied homes in $1000's

'''

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('csv folder/housing.csv', header=None, delimiter=r"\s+", names=column_names)
data = data.dropna()


'''
Correlation Plot
'''
f = plt.figure(figsize=(15, 18), dpi=50)
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=90)
plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.savefig('images_Housing/Corr matrix Housing.png')
plt.show()
plt.close()

'''
Thanks to the result of the plot, I choose the attributes that correlate the most,
e.g. 'LSTAT' and 'MEDV'
'''

X = data['LSTAT'].values
X = X.reshape(-1, 1)
y = data['MEDV'].values
Y = y.reshape(-1, 1)

'''
Splitting into train and test
'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler=StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

'''
Execution times of the 3 algorithms
'''

print("==============GD==============")
start_gd = datetime.now()
model_gd = MySGD(learning_rate = 0.001)
history_gd, _ = model_gd.fit(X_train, y_train, batch_size=1,epochs=100)
end_gd=datetime.now() - start_gd
print('Time: ', end_gd) 
predictions_gd_test = model_gd.predict(X_test)
predictions_gd_train = model_gd.predict(X_train)

print("==============SGD==============")
start_sgd = datetime.now()
model_sgd = MySGD(learning_rate = 0.001)
history_sgd, _ = model_sgd.fit(X_train, y_train, batch_size = 32, epochs = 100)
end_sgd=datetime.now() - start_sgd
print('Time: ', end_sgd) 
predictions_sgd_test = model_sgd.predict(X_test)
predictions_sgd_train = model_sgd.predict(X_train)

print("=============ADAM=============")
start_adam = datetime.now()
model_adam = MyAdam(learning_rate = 0.01)
history_adam, _ = model_adam.fit(X_train, y_train, batch_size = 32, epochs = 100)
end_adam=datetime.now() - start_adam
print('Time: ',end_adam ) 
predictions_adam_test = model_adam.predict(X_test)
predictions_adam_train = model_adam.predict(X_train)

'''
Animation of the losses plot for GD, SGD, ADAM
'''
x = np.arange(1, 101)
fig, ax = plt.subplots()
l1, = ax.plot([], [], 'o-', color="red", label='gd', markevery=[-1])
l2, = ax.plot([], [], 'o-', color="blue",label='sgd', markevery=[-1])
l3, = ax.plot([], [], 'o-', color="green",label='adam', markevery=[-1])
ax.legend(loc='center right')
ax.set_xlim(0,100)
ax.set_ylim(0.01, 1.3)
ax.set_xlabel('Epochs')
ax.set_ylabel('Cost')

def animate(i):
    l1.set_data(x[:i], history_gd[:i])
    l2.set_data(x[:i], history_sgd[:i])
    l3.set_data(x[:i], history_adam[:i])
    return (l1,l2)

ani = animation.FuncAnimation(fig, animate, frames=100, interval=10)
plt.show()

'''
Losses for GD, SGD, ADAM
'''
x=np.arange(1, 101)
plt.plot(x, history_gd, color='red',label='gd')
plt.plot(x, history_sgd, color='blue', label='sgd')
plt.plot(x, history_adam, color='green',label='adam')
plt.title("Losses")
plt.legend(loc='upper right')
plt.savefig('images_Housing/Losses_Housing.png')
plt.show()


'''
Losses for GD, SGD, ADAM with x-axis shared in 3 different subplots
'''
fig, axs=plt.subplots(3)
x=np.arange(1, 101)
axs[0].plot(x, history_gd, color='red')
axs[0].set_title('GD Loss')
axs[1].plot(x, history_sgd, color='blue')
axs[1].set_title('SGD Loss')
axs[2].plot(x, history_adam, color='green')
axs[2].set_title('Adam Loss ')
fig.tight_layout()
plt.savefig('images_Housing/Losses with x-axes shared_Housing.png')
plt.show()


'''
GD (batch_size=1) vs SGD: demonstration that the timing decreases as the batch_size increases
'''
batchs=[1, 32, 64, 128, 256]
times_gd_sgd=[]

for b in batchs:
    
    start_sgd = datetime.now()
    model_sgd = MySGD(learning_rate = 0.001)
    history_sgd, _ = model_sgd.fit(X_train, y_train, batch_size = b, epochs = 100)
    end_sgd=datetime.now() - start_sgd
    time_sgd=end_sgd.total_seconds()*1000
    times_gd_sgd.append(time_sgd)

plt.plot(batchs, times_gd_sgd)
plt.scatter(batchs, times_gd_sgd)
plt.title('Batch vs Time for GD vs. SGD')
plt.xlabel("Batch size")
plt.ylabel("Time in ms")
fig.tight_layout()
plt.savefig('images_Housing/Timing gd vs sgd_Housing')
plt.show()


'''
ADAM with different parameters of learning rate and batch_size
'''
learning_rate=[1e-4, 1e-3, 1e-2] 
momentum_decay_12 = [0.9, 0.999]
batchs=[32, 64, 128, 256]
epsilon = [10 ** -8]
plot_list=[]
for lr in learning_rate:
    for bs in batchs:
        for e in epsilon:
            model_adam = MyAdam(learning_rate = lr, momentum_decay=momentum_decay_12, epsilon=e)
            history_adam, _ = model_adam.fit(X_train, y_train, batch_size = bs, epochs = 1000)
            a=plt.plot(np.arange(1, 1001), history_adam, label='Batch size ={}'.format(bs))
            plt.title("Adam loss with lr_size={}, eps={}".format(lr, e))
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
    plt.legend()
    plt.savefig('images_Housing/adam parameters with lr={}_Housing.png'.format(lr))
    plt.show()


'''
MSE
'''

fig, axs=plt.subplots(3)
axs[0].plot(y_test, label='Actual')
axs[0].plot(predictions_gd_test, label='Predicted')
mes_gd=mean_squared_error(y_test, predictions_gd_test)
axs[0].set_title('MSE GD= {}'.format(mes_gd))

axs[1].plot(y_test, label='Actual')
axs[1].plot(predictions_sgd_test, label='Predicted')
mes_sgd=mean_squared_error(y_test, predictions_sgd_test)
axs[1].set_title('MSE SGD= {}'.format(mes_sgd))

axs[2].plot(y_test, label='Actual')
axs[2].plot(predictions_adam_test, label='Predicted')
mes_adam=mean_squared_error(y_test, predictions_adam_test)
axs[2].set_title('MSE ADAM= {}'.format(mes_adam))

plt.legend()
fig.tight_layout()
plt.savefig('images_Housing/MSE_Housing.png')
plt.show()

'''
Regression Curve
'''
fig, axs=plt.subplots(3, 2)
fig.suptitle('Regression curve on training (on the left) and test set (on the right)')
axs[0,0].scatter(X_train, y_train, marker='o', color='m',s=30)
axs[0,0].plot(X_train, predictions_gd_train)
axs[0,0].set_title('GD')
axs[1,0].scatter(X_train, y_train, marker='o', color='m',s=30)
axs[1,0].plot(X_train, predictions_sgd_train)
axs[1,0].set_title('SGD')
axs[2,0].scatter(X_train, y_train, marker='o', color='m',s=30)
axs[2,0].plot(X_train, predictions_adam_train)
axs[2,0].set_title('ADAM')

axs[0,1].scatter(X_test, y_test, marker='o', color='m',s=30)
axs[0,1].plot(X_test, predictions_gd_test)
axs[0,1].set_title('GD')
axs[1,1].scatter(X_test, y_test, marker='o', color='m',s=30)
axs[1,1].plot(X_test, predictions_sgd_test)
axs[1,1].set_title('SGD')
axs[2,1].scatter(X_test, y_test, marker='o', color='m',s=30)
axs[2,1].plot(X_test, predictions_adam_test)
axs[2,1].set_title('ADAM')

fig.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('images_Housing/Regression curve_Housing.png')
plt.show()


'''
Early stopping
'''
print("==============GD==============")
model_gd = MySGD(learning_rate = 0.001)
history_gd, n_iter_gd = model_gd.fit(X_train, y_train, batch_size=1,epochs=100, early_stopping=True, thr=0.0001)
print("GD n_iter: " + str(n_iter_gd))

print("==============SGD==============")
model_sgd = MySGD(learning_rate = 0.001)
history_sgd, n_iter_sgd = model_sgd.fit(X_train, y_train, batch_size = 32, epochs = 100, early_stopping=True, thr=0.0001)
print("SGD n_iter: " + str(n_iter_sgd))

print("=============ADAM=============")
model_adam = MyAdam(learning_rate = 0.01)
history_adam, n_iter_adam = model_adam.fit(X_train, y_train, batch_size = 32, epochs = 100, early_stopping=True, thr=0.0001)
print("ADAM n_iter: " + str(n_iter_adam))
