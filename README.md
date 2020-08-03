# neuralnet
Implementing a neural network by hand

In this project, I manually implemented a Neural Network model, using Stochastic Gradient Descent to fit in the Santander Customer Experience dataset.
I used LogLoss as the cost function, and backpropagation to upload the gradient, respect to the weights and bias.
The Network has only one hidden layer, using ReLu activation, and the output uses the sigmoid as activation function to give a binary response.
The bench model was Xgboost with 0.86 AUC performance. This model achives 0.85 AUC. It may be worth it to add more hidden layers.
