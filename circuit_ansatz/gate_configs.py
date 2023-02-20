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
        self.all_gate_params = {}
        self._gate_id = 0
            
        if num_of_clbits is None:
            num_of_clbits = num_of_qubits
        self._qc = QuantumCircuit(num_of_qubits, num_of_clbits)
        
    def build_cascade_hadamard(self):
        for idx in range(self._num_of_qubits):
            self._qc.h(idx)
    
    def build_cascade_rx(self):
        params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        for idx in range(self._num_of_qubits):
            self._qc.rx(params[idx], idx)
            self._gate_id += 1
            self.all_gate_params['rx'+str(self._gate_id)] = params[idx]
            
    def build_cascade_rz(self):
        params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        for idx in range(self._num_of_qubits):
            self._qc.rz(params[idx], idx)
            self._gate_id += 1
            self.all_gate_params['rz'+str(self._gate_id)] = params[idx]
            
    def cx_all_neighbors(self):
        for i in range(self._num_of_qubits-1):
            control = i
            target = i+1
            self._qc.cx(control, target)
        self._qc.cx(self._num_of_qubits-1, 0)
    
    def cx_one_to_all(self):       
        index_list = [i for i in range(self._num_of_qubits)]
        
        for j in range(self._num_of_qubits):
            control = j
            temp_index_list = index_list.copy()
            temp_index_list.remove(control)
            for target in temp_index_list:
                self._qc.cx(control, target)
    
    
    
    
    
    
    
    
    
    
    
                