import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
from quantum_model import x, y_, loss, optimizer

CKPT_DIR = './save'
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
LOG_PATH = './logs'
summary_writer = tf.summary.FileWriter(LOG_PATH, graph=tf.get_default_graph())

epochs = 30
batch_size = 100
sess.run(init)

# Train over the dataset for 30 epochs
for epoch in range(epochs):
    for i in range(driving_data.num_images // batch_size):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        _, loss_value = sess.run([optimizer, loss], feed_dict={x: xs, y_: ys})
        print(f"Epoch: {epoch}, Step: {epoch * batch_size + i}, Loss: {loss_value}")

        # Write logs at every iteration
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=loss_value)
        summary_writer.add_summary(summary, epoch * driving_data.num_images / batch_size + i)

        # save a checkpoint every batch
        if i % batch_size == 0:
            if not os.path.exists(CKPT_DIR):
                os.makedirs(CKPT_DIR)
            checkpoint_path = os.path.join(CKPT_DIR, "sample_circuit9.ckpt") #the file name must be changed every time we run with a new circuit 
            filename = saver.save(sess, checkpoint_path)

    print("Model saved in file: %s" % filename)

print("Run the command line:")
print("--> tensorboard --CKPT_DIR=./logs")
print("Then open http://0.0.0.0:6006/ in your web browser")


