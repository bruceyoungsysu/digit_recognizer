import tensorflow as tf

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('ckpt/model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))

    graph = tf.get_default_graph()
    print(graph.get_collection('variables'))
    # w = graph.get_tensor_by_name('total_loss:0')