import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

# 하이퍼 파라미터 설정
Learning_rate = 0.0004
training_epoch = 20
n_hidden = 256
n_input = 28*28
batch_size = 100

# 입력 x의 플레이스 홀더 설정
X = tf.placeholder(tf.float32, [None, n_input])

# Encoder
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# Decoder
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

# Q. loss function을 구현하세요.
cost = tf.reduce_mean(tf.pow(X - decoder, 2))

# Q. Optimizer를 구현하세요
optimizer = tf.train.RMSPropOptimizer(Learning_rate).minimize(cost)

# Optimization function
sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(training_epoch):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost],
                           feed_dict={X: batch_xs})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
      'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))
print("train_finsh")

# 결과값 확인
# 디코더로 생성한 이미지를 직접 확인
# Fig.a : 학습 이미지 아래 : Fig.b : 디코더 생성 이미지
sample_size = 10
samples = sess.run(decoder,feed_dict={X: mnist.test.images[:sample_size]})
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
plt.show()