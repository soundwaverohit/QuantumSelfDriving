# circuit_6.py

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble, execute, QuantumRegister, ClassicalRegister
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator

# Importing other libraries
import numpy as np
import random

class circuit:   
    def __init__(self, num_of_qubits, num_of_clbits=None, list_of_parameters=None):
        self._num_of_qubits = num_of_qubits
        self._gate_id = 0
        self.all_gate_params = {}
        
        if num_of_clbits is None:
            num_of_clbits = num_of_qubits
        self._qc = QuantumCircuit(num_of_qubits, num_of_clbits)
        
    def build_cascade_hadamard(self):
        for idx in range(self._num_of_qubits):
            self._qc.h(idx)
            self._gate_id += 1
            #self.all_gate_params['h'+str(idx)] = ''
    
    def build_cascade_rx(self):
        params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        for idx in range(self._num_of_qubits):
            self._qc.rx(params[idx], idx)
            self._gate_id += 1
            self.all_gate_params['rx'+str(idx)] = params[idx]
            
    
if __name__ == '__main__':
    qc = circuit(4)
    qc.build_cascade_hadamard()
    qc.build_cascade_rx()
    print(qc._qc.draw())
    print(qc.all_gate_params)