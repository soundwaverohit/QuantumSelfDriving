# INtructions on creating different different quantum circuits 


## 1. Go to the File quantum_model.py which has the Classical CNN and the Variational Quantum Circuit.


## 2. Implement a different Variational Quantum Circuit in the def variational_quantum_circuit(inputs, weights): function 


## 3. Go to quant_train.py file and change the checkpoint_path = os.path.join(LOGDIR, "sample_circuit1.ckpt") line 36, change the "sample_circuit1.ckpt" with a different name each time you are about to train.


## 4. Go to cd Quantum_Model directory and run "python3 quant_train.py" to train that model


## 5: To test the model once it is trained go to the run_dataset.py and in saver.restore(sess, "save/sample_circuit1.ckpt") change the model name to the one saved and run "python3 run_dataset.py" to evaluate its performance.

