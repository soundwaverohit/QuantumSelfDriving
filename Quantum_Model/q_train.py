import os
import tensorflow.compat.v1 as tf
tf.config.run_functions_eagerly(True)

tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
from CNN_Model import driving_data
from CNN_Model import model
import pennylane as qml
from pennylane import numpy as np

LOGDIR = './save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()


loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# Convert TensorFlow weights to PennyLane parameters
@tf.function
def get_pennylane_weights(tf_weights):
    pennylane_weights = []
    sess.run(tf.global_variables_initializer())  # Initialize TensorFlow variables
    for weight in tf_weights:
        weight_value = sess.run(weight)
        pennylane_weights.append(np.arcsin(weight_value))
    return pennylane_weights


# Convert PennyLane parameters to TensorFlow weights
@tf.function
def get_tf_weights(pennylane_weights):
    #tf_weights = []
    #for weight in pennylane_weights:
    #    tf_weights.append(np.sin(weight))

    tf_weights= tf.convert_to_tensor(pennylane_weights)
    return tf_weights

# Create placeholders for inputs and labels
x = tf.placeholder(tf.float32, shape=[100, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[100, 1])

print("YYYYYYYYYYYYY: ", y_)

# Replace the weights in the TensorFlow model with the quantum circuit parameters
pennylane_weights = get_pennylane_weights(tf.trainable_variables())
pennylane_weights_np = np.array(pennylane_weights, dtype=object)
pennylane_weights_bytes = pennylane_weights_np.tobytes()

#pennylane_weights_tensor = tf.Variable(pennylane_weights_np, trainable=True)
pennylane_weights_tensor = tf.Variable(pennylane_weights_bytes, trainable=True)


tf_weights = get_tf_weights(pennylane_weights_tensor)
output = model.y

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 30
batch_size = 100

# train over the dataset about 30 times
for epoch in range(epochs):
    for i in range(int(driving_data.num_images/batch_size)): 
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        feed_dict = {model.x: xs, model.y_: ys, model.keep_prob: 1.0}
        train_step.run(feed_dict= feed_dict)
        #train_step.run(feed_dict={x: xs, y_: ys, model.keep_prob: 1.0})
        if i % 10 == 0:
            xs, ys = driving_data.LoadValBatch(batch_size)
            loss_value = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

        # write logs at every iteration
        summary = merged_summary_op.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "quantum_model1.ckpt")
            filename = saver.save(sess, checkpoint_path)

    print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
