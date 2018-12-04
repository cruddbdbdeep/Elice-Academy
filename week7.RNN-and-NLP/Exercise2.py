def main():
    # RNN을 이용해 MNIST 분류를 해결하는 문제입니다.
    import tensorflow as tf

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    ############
    # 옵션 설정
    ############
    learning_rate = 0.001
    total_epoch = 5
    batch_size = 128

    # RNN은 seauential data를 다루며,
    # 문제설명처럼 28x28 mnist 데이터의 28x1 row vector 총 28개가 하나의 데이터입니다.
    # 이에 따라 한 번에 입력받는 갯수 n_input과 RNN의 hidden layer의 갯수 n_step을 설정해야합니다.
    # 이를 위해 가로 픽셀수를 n_input으로, 세로 픽셀수를 입력 단계인 n_step으로 설정해야합니다.
    n_input = 28
    n_step = 28
    n_hidden = 128
    n_class = 10

    ###################
    # 신경망 모델 구성
    ###################

    # input와 output을 설정합니다.
    X = tf.placeholder(tf.float32, [None, n_step, n_input])
    Y = tf.placeholder(tf.float32, [None, n_class])

    # output을 생성하기 위한 weight W_hy와 bias b_hy를 설정합니다.
    # Q1. output의 dimension을 고려하여 W_hy와 b_hy 노드를 생성하세요.
    W_hy = tf.Variable(tf.random_normal([n_hidden, n_class]))
    b_hy = tf.Variable(tf.random_normal([n_class]))

    # 아래 과정은 RNN 셀을 생성하는 과정으로 순차적으로 따라와주시면 됩니다!!!!!
    # 아래 함수를 사용하면 여러 종류의 RNN을 설정할 수 있고 여기서는 BasicRNNCell을 사용하겠습니다.
    # BasicRNNCell,BasicLSTMCell,GRUCell
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

    # RNN 신경망을 생성합니다.
    # CNN 의 tf.nn.conv2d처럼 아래 tf.nn.dynamic_rnn 함수를 사용하면
    # n_step의 hidden state를 생성하는 과정을 한번에 해결할 수 있습니다!
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # 위에서 얻은 outputs(hidden state)는 [batch_size, n_step, n_hidden]의 shape를 가집니다.
    # 해당 outputs(hidden state)를 이용해 [batch_size, n_class] shape를 가지는 최종 output Y를 생성하기 위해
    # 해당 outputs의 shape를 [batch_size, n_hidden]로 바꿔야합니다.
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]

    # Q2. 최종 output인 pred를 생성하는 노드를 생성해 해주세요.
    pred = tf.matmul(outputs, W_hy) + b_hy

    # Q3. cross entropy를 이용해 RNN의 예측 cost를 생성해주세요.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

    # Q4. cost를 최소화하는 Adam optimizer를 생성해주세요.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    ###################
    # 신경망 모델 학습
    ###################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Q5. total batch size를 계산해주세요.
    total_batch = int(mnist.train.num_examples / batch_size)
    print('batch :', int(mnist.train.num_examples))

    for epoch in range(total_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환합니다.
            batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

            _, cost_val, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1),
              'Avg cost =', '{:.3f}'.format(total_cost / total_batch),
              'Train acc = ', '{:.3f}'.format(acc))

    test_batch_size = len(mnist.test.images)
    test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
    test_ys = mnist.test.labels
    test_acc = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys})

    return test_acc

if __name__ == "__main__":
    main()