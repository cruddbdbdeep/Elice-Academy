import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# seed값 설정
tf.set_random_seed(1234)

# 데이터 불러오기
mnist = input_data.read_data_sets("./data/", one_hot=True)

# Hyper-parameters
epochs = 100
learning_rate = 0.001
batch_size = 128

# Q. MNIST 데이터를 받을 수 있게 placeholder를 설정 해주세요.
# MNIST 사진 한장의 크기는 28 x 28이며, 정답은 0~9 사이의 자연수입니다.
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Q. Input layer -> hidden-1 layer 단계의 weight / bias Variable를 만들어주세요.
# hidden-1 layer의 dimension은 1000으로 설정해주세요.
# 초기값 설정은 자유롭게 해주시면 됩니다.
W1 = tf.Variable(tf.random_normal(shape=[784, 1000]))
b1 = tf.Variable(tf.random_normal(shape=[1000]))

# Q. hidden-1 layer -> hidden-2 layer 단계의 weight / bias Variable를 만들어주세요.
# hidden-2 layer의 dimension은 1000으로 설정해주세요.
# 초기값 설정은 자유롭게 해주시면 됩니다.
W2 = tf.Variable(tf.random_normal(shape=[1000, 1000]))
b2 = tf.Variable(tf.random_normal(shape=[1000]))

# Q. hidden-2 layer -> output layer 단계의 weight / bias Variable를 만들어주세요.
# 초기값 설정은 자유롭게 해주시면 됩니다.
W3 = tf.Variable(tf.random_normal(shape=[1000, 10]))
b3 = tf.Variable(tf.random_normal(shape=[10]))

# Q. hidden-1 layer의 값을 연산하는 node를 만들어주세요.
# activation function은 relu를 사용하세요.
hidden_1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# Q. hidden-2 layer의 값을 연산하는 node를 만들어주세요.
# activation function은 relu를 사용하세요.
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W2) + b2)

# Q. output layer의 값을 연산하는 node를 만들어주세요.
logit = tf.matmul(hidden_2, W3) + b3

# loss와 optimizer를 설정해주는 단계입니다.
# Q. 아래 비어있는 input을 채워주세요.
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y)) # input 필요
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # input 필요
train_op = optimizer.minimize(loss_op) # input 필요

# 여기서 부터는 수정하지 마세요.
correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % 5 == 0:
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Training loss= " + "{:.4f}".format(loss))

    print("Optimization Finished!")
    acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print('Test accuracy:', acc)