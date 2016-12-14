import tensorflow as tf  															"""tensorflow를 import함"""
from tensorflow.examples.tutorials.mnist import input_data						

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)			"""mnist 데이터 다운로드"""
//
# Set up model
x = tf.placeholder(tf.float32, [None, 784])											"""MNIST 784차원의 벡터로 단조화된 이미지들에 어떤 숫자들이든 입력할 수 있기위해 
																					   [None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현"""
W = tf.Variable(tf.zeros([784, 10]))												"""W를 0으로 채워진 텐서들로 초기화"""
b = tf.Variable(tf.zeros([10]))														"""b를 0으로 체워진 텐서들로 초기화"""
y = tf.nn.softmax(tf.matmul(x, W) + b)												"""tf.matmul(x, W)로x와 W를 곱함,그 다음 b를 더하고,
																					   tf.nn.softmax 를 적용"""

y_ = tf.placeholder(tf.float32, [None, 10])											"""정답을 입력하기 위한 새 placeholder를 추가함"""

cross_entropy = -tf.reduce_sum(y_*tf.log(y))										"""교차 엔트로피 −∑y′log(y)를 구현"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)		"""TensorFlow에게 학습도를 0.01로 준 경사 하강법(gradient descent)
																					   알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령"""

# Session
init = tf.initialize_all_variables()												"""변수들을 초기화"""

sess = tf.Session()
sess.run(init)																		"""세션에서 모델을 시작"""

# Learning
for i in range(1000):																"""1000번 반복 즉, 1000번 학습"""
  batch_xs, batch_ys = mnist.train.next_batch(100)									"""학습 세트로부터 100개의 무작위 데이터들의 배치들을 가져옴"""
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})						"""placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩"""

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))						"""tf.equal 을 이용해 예측이 실제와 맞았는지 확인"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))					"""부정소숫점으로 캐스팅한 후 평균값을 구함"""

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))	"""정확도 출력"""