#--------------------------------------------------------------------------------
# Graph 및 Session
# page 299
#--------------------------------------------------------------------------------
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2
print('\nResult = {}\n\n'.format(f))

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print('\n[Basic structure]f = {}\n\n'.format(result))
sess.close()

with tf.Session() as sess:
    x.initializer.run()   #  tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = sess.run(f)      #  result = tf.get_default_session().run(f)
    print('\n[with 구문]f = {}\n\n'.format(result))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = sess.run(f)
    print('\n[global initialization]f = {}\n\n'.format(result))
#--------------------------------------------------------------------------------
# InteractiveSession()
# page 301
#--------------------------------------------------------------------------------
sess = tf.InteractiveSession()
init.run()
#result = sess(f)
result = f.eval()
print('\n[InteractiveSession]f = {}\n\n'.format(result))

#--------------------------------------------------------------------------------
# 9.3 계산 그래프 관리
# page 301
#--------------------------------------------------------------------------------
x1 = tf.Variable(1)
print('\nx1.graph = tf.get_default_graph(): {}\n\n'.format(x1.graph is tf.get_default_graph()))
print('\nx1.graph \n={}'.format(x1.graph))
print('\nx1 Node definitions = {}\n\n'.format(x1.op.node_def))
print('\nx1._variable = {}\n'.format(x1._variable))
print('\nx1 Node definitions(x1_variable) = {}\n\n'.format(x1._variable.op.node_def))
print('\nTotal Graph(): (variable) = {}\n\n'.format(x1.graph.as_graph_def()))

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
print('\nx2.graph is graph? {}\n\n'.format(x2.graph is graph))
print('\nx2.graph is tf.get_default_graph()? {}\n\n'.format(x2.graph is tf.get_default_graph()))

#--------------------------------------------------------------------------------
# 9.4 노드값의 생애주기
# page 302
#--------------------------------------------------------------------------------
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print('\ny = {}\n\n'.format(y.eval())) # print('\ny = {}\n\n'.format(sess.run(y)))
    print('\nz = {}\n\n'.format(z.eval())) # print('\nz = {}\n\n'.format(sess.run(z)))

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print('\ny = {}\n\n'.format(y.eval()))
    print('\nz = {}\n\n'.format(z.eval()))

#--------------------------------------------------------------------------------
# 9.5 텐서플로를 이용한 선형회귀
# page 303
#--------------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print('\nOptimal theta(normal equation) = \n{}\n\n'.format(theta_value))

#--------------------------------------------------------------------------------
# 9.6 경사하강법 구현
# page 305
#--------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 ==0:
            print('Epoch = {}, MSE = {}'.format(epoch, mse.eval()))
        sess.run(training_op)

    best_theta = theta.eval()
    print('\nOptimal Theta(Gradient) = \n{}\n\n'.format(best_theta))

#--------------------------------------------------------------------------------
# 9.6.2 자동미분사용
# page 306
#--------------------------------------------------------------------------------
tf.reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 직접계산: gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)

     for epoch in range(n_epochs):
         if epoch % 100 == 0:
             print('Epoch= {}, MSE = {}'.format(epoch, mse.eval()))
         sess.run(training_op)
     best_theta = theta.eval()
     print('\nOptimal Theta(Gradient: symbolic diff.) = \n{}\n\n'.format(best_theta))

#--------------------------------------------------------------------------------
# 9.6.3 Optimizer 사용
# page 308
#--------------------------------------------------------------------------------
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Optimizer 호출부분
training_op = optimizer.minimize(mse)   # Optimizer 적용

init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     for epoch in range(n_epochs):
         if epoch % 100 == 0:
             print("Epoch", epoch, "MSE =", mse.eval())
         sess.run(training_op)
     best_theta = theta.eval()
     print('\nOptimal Theta(Gradient: Optimizer) = \n{}\n\n'.format(best_theta))

#--------------------------------------------------------------------------------
# 9.7 placeholder
# page 308
#--------------------------------------------------------------------------------

A = tf. placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    #B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
    #B_val_2 = B.eval(feed_dict={A: [[4,5,6], [7,8,9]]})
    B_val_1 = sess.run(B, feed_dict={A:[[1,2,3]]})
    B_val_2 = sess.run(B, feed_dict={A: [[4,5,6], [7,8,9]]})
    print('\nB_val_1 = \n{}\n\n'.format(B_val_1))
    print('\nB_val_2 = \n{}\n\n'.format(B_val_2))

#--------------------------------------------------------------------------------
# placeholder를 이용한 Gradient Descent
# page 309
#--------------------------------------------------------------------------------

tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Visualizing the graph

with tf.Session() as sess:
     sess.run(init)

     for epoch in range(n_epochs):
         if epoch % 100 == 0:
             print("Epoch", epoch, "MSE =", mse.eval())
             save_path = saver.save(sess, "/tmp/my_model")
         sess.run(training_op)

     best_theta = theta.eval()
     save_path = saver.save(sess, "my_model_final.ckpt")

print("Best theta:")
print(best_theta)
