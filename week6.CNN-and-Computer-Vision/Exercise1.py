import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

dataname = 'mnist'
mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

## 하이퍼 파라미터 설정
total_epoch = 100
batch_size = 100
Learning_rate = 0.0002
n_hidden1 = 256
n_input = 28 * 28
n_noise = 128

# 플레이스 홀더 설정
X = tf.placeholder(tf.float32, [None, n_input])     # MNIST = 28*28
Z = tf.placeholder(tf.float32, [None, n_noise])     # Noise Dimension = 128

# Generator Network
# Q. Generator input layer -> hidden-1 layer 단계의 weight / bias variable를 만들어주세요.
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden1], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden1]))
# Q. Generator hidden1 layer -> output layer 단계의 weight / bias variable를 만들어주세요.
G_W2 = tf.Variable(tf.random_normal([n_hidden1, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# Discriminator Network
# Q. Discriminator input layer -> hidden-1 layer 단계의 weight / bias variable를 만들어주세요.
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden1]))
# Q. Discriminator hidden-1 layer -> output layer 단계의 weight / bias variable를 만들어주세요
D_W2 = tf.Variable(tf.random_normal([n_hidden1, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# Generator
def generator(noise_z):
    hidden1 = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden1, G_W2) + G_b2)
    return output

# Discriminator
def discriminator(inputs):
    hidden1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden1, D_W2) + D_b2)
    return output

# 무작위 노이즈 생성
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 생성자/구분자 순서
# Q. 아래 비어있는 함수를 불러와 실행해 주세요.
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# Loss function
# Q. Q. 최대화 해야 하는 loss 설정해주세요.
loss_D = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = -tf.reduce_mean(tf.log(D_gene))

# training
# D 학습시에는 D만 학습되도록
# G 학습 시에는 G만 학습되도록
# 아래 리스트에 판별자 가중치와 바이어스를 선언하세요.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 학습 알고리즘 설정
# 최소화를 위해 loss 를 음수화
train_D = tf.train.AdamOptimizer(Learning_rate).minimize(loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(Learning_rate).minimize(loss_G, var_list=G_var_list)
# 학습 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples / batch_size)
saver = tf.train.Saver()

loss_val_D, loss_val_G = 0, 0
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
    print("Epoch:", "%04d" % epoch, 'D_loss:{:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    # iteration 10마다 save
    if (i%10)==0:
         # local에서 파일 생성 경우 result/후 gan_데이터셋이름을 추가해서 미리 선언하세요.
         saver.save(sess, './result/gan_'+dataname+'/my-model', global_step=i, write_meta_graph=False)

# 학습 결과 학인
if epoch ==0 or (epoch+1)%10 ==0:
    sample_size = 10
    noise = get_noise(sample_size,n_noise)

    samples = sess.run(G, feed_dict={Z:noise})
    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
    plt.savefig('./result/gan_'+dataname+'/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
print('finish training')