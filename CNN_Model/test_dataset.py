import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
from subprocess import call
import os
import driving_data
import pandas as pd

#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/modellong2.ckpt")


smoothed_angle = 0
i = 0


batch_size = 10000
xs, ys = driving_data.LoadValBatch(batch_size)

print(ys[1])

yss=[]
for nums in ys:
    yss.append(int(nums[0]))

arrs= []

for i in range(len(xs)):
    full_image = cv2.imread("../Images/driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    arrs.append(int(degrees))


num_matches = sum(arr == y for arr, y in zip(arrs, yss))
percentage_match = (num_matches / len(arrs)) * 100

print(f"Model Performance: {percentage_match}%")