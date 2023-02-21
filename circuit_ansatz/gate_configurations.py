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
    def __init__(self, num_of_qubits, num_of_clbits=None): 
        self._num_of_qubits = num_of_qubits 
        self._all_gate_params = {} 
        self._gate_id = 0 
            
        if num_of_clbits is None: 
            num_of_clbits = num_of_qubits 
             
        self._qc = QuantumCircuit(num_of_qubits, num_of_clbits)
        
    def get_quantum_circuit(self):
        return self._qc
    
    def get_rotation_params(self):
        return self._all_gate_params
    
    def build_cascade_hadamard(self): 
        for idx in range(self._num_of_qubits): 
            self._qc.h(idx)
    