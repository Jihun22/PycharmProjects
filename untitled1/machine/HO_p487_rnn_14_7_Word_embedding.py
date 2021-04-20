import tensorflow as tf
#----------------------------------------------------------------
#  RNN NSP(Natural Language Processing)
#  page 515
#----------------------------------------------------------------
vocabulary_size = 50000
embedding_size = 150

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0,0.0)
embeddings = tf.Variable(init_embeds)

train_inputs = tf.placeholder(tf.float32, shape=[None])  # 식별자를 주입
embed = tf.nn.embedding_lookup(embeddings, train_inputs) #임베딩으로 변환
