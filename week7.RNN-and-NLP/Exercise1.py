# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무소녀놀이사랑공책<>']
word2idx = {word: idx for idx, word in enumerate(char_arr)}

seq_data = [['word', '단어'], ['wood', '나무'],
            ['play', '놀이'], ['girl', '소녀'],
            ['love', '사랑'], ['note', '공책']]
# hyperparams
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
n_class = n_input = len(word2idx)


# Batch data 생성
# <S> : 디코딩 입력의 시작을 나타내는 기호
# <E> : 디코딩 출력의 끝을 나타내는 기호
# <P> : 배치 데이터의 time step 보다 크기가 작은 경우 빈 시퀀스의 길이를 채우는 기호
def getBatch(seq_data):
    input_batch = [];
    output_batch = [];
    target_batch = []
    pos = 0;
    word_len = len(word2idx)
    while pos < len(seq_data):
        seq = seq_data[pos]
        # 영어를 한글로 바꾸는 task이므로 seq의 첫번째 원소인 영어를 넣어준다.
        input_word = [word2idx[i] for i in seq[0]]
        ###############################################
        #####output_word와 target_word를 작성하시오#####
        # input_word와 비슷하게 작성한다.
        # output_word : seq의 두번째 원소인 한글에 대해서 <S> 기호를 처음에 추가한다. -> decoder cell의 입력
        # target_word : seq의 두번째 원소인 한글에 대해서 <E> 기호를 마지막에 추가한다. -> decoder cell의 출력
        output_word = [word2idx[i] for i in ('S' + seq[1])]
        target_word = [word2idx[i] for i in (seq[1]) + 'E']
        ###############################################
        input_batch.append(np.eye(word_len)[input_word])
        output_batch.append(np.eye(word_len)[output_word])
        target_batch.append(target_word)
        pos += 1
    return input_batch, output_batch, target_batch


# Encoder - Decoder network
# 신경망 모델 구성
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
tf.reset_default_graph()
enc_input = tf.placeholder(tf.float32, [None, None, n_input])  # [batch size, time steps, input size]
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])  # [batch size, time steps]

# Encoder cell 작성
with tf.variable_scope('encode'):
    # tf.nn.rnn_cell.BasicRNNCell 를 사용해서 enc_cell 구성
    # tf.nn.rnn_cell.DropoutWrapper 을 사용해서 dropout 적용, dropout 비율은 0.5로 지정
    #######################################################
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    #######################################################
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# Decoder cell 작성
with tf.variable_scope('decode'):
    #######################################################
    #####위의 encoder cell 작성과 동일하게 decoder 구성#####
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    # Seq2Seq 모델은 인코더 셀의 최종 상태값을 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    # tf.nn.dynamic_rnn을 이용하되 위의 encoder cell과 다르게 decoder 셀의 초기값을 고려
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)
    #######################################################
model = tf.layers.dense(outputs, n_class, activation=None)
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Train network
# 신경망 모델 학습
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = getBatch(seq_data)
#######################################################
# feed_dict 작성 : input_batch, output_batch, target_batch를 적절하게 넣기
for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})
    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.4f}'.format(loss))
print('model training finish!')

#######################################################
saver.save(sess, './model/seq2seq.ckpt', global_step=total_epoch)


# Test model
# seq2seq를 이용해서 실제로 번역되는 결과 확인
def translate(word):
    # 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = getBatch([seq_data])
    ##########################################################
    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    # tf.argmax를 사용해서 model에 적용하며 위의 train과 동일하게 feed_dict 작성
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    ##########################################################
    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]
    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    return translated


def main():
    print('word ->', translate('word'))
    print('wood ->', translate('wood'))
    print('play ->', translate('play'))
    print('love ->', translate('love'))
    print('girl ->', translate('girl'))
    print('note ->', translate('note'))

    result = list();
    accuracy = list();
    answer = seq_data
    target = ['단어', '나무', '놀이', '소녀', '사랑', '공책']
    for source in ['word', 'wood', 'play', 'girl', 'love', 'note']:
        result.append(translate(source))

    for pos, (res, tar) in enumerate(zip(result, target)):
        accuracy.append([res, target] == answer)

    return sum(accuracy) / len(accuracy)


# test
if __name__ == "__main__":
    main()