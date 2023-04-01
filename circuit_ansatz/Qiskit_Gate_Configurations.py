# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble, execute, QuantumRegister, ClassicalRegister
from qiskit.visualization import *

# Importing other libraries
import numpy as np
import random

class circuit:   
    def __init__(self, num_of_qubits, rotation_params=None, num_of_clbits=None): 
        self._num_of_qubits = num_of_qubits 
        
        self._all_gate_params = {}
        if rotation_params != None:
            self._all_gate_params = rotation_params 
        
        self._gate_id = 0 
        
        if num_of_clbits is None: 
            num_of_clbits = num_of_qubits 
             
        self._qc = QuantumCircuit(num_of_qubits, num_of_clbits)
        
    def get_quantum_circuit(self):
        return self._qc
    
    def get_rotation_params(self):
        return self._all_gate_params
    
    # applies the hadamard gate to every qubit
    def build_cascade_hadamard(self): 
        for idx in range(self._num_of_qubits): 
            self._qc.h(idx)
            
    # applies the rx gate to every qubit        
    def build_cascade_rx(self, params=None):
        if params != None:
            for idx in range(self._num_of_qubits):
                self._qc.rx(params[idx], idx)
                self._all_gate_params['w'+str(self._gate_id)] = params[idx]
                self._gate_id += 1
        else:
            for idx in range(self._num_of_qubits):
                self._qc.rx(self._all_gate_params['w'+str(self._gate_id)], idx)
                self._gate_id += 1
            
    # applies the rz gate to every qubit        
    def build_cascade_rz(self, params=None):
        if params != None:
            for idx in range(self._num_of_qubits):
                self._qc.rz(params[idx], idx)
                self._all_gate_params['w'+str(self._gate_id)] = params[idx]
                self._gate_id += 1
        else:
            for idx in range(self._num_of_qubits):
                self._qc.rz(self._all_gate_params['w'+str(self._gate_id)], idx)
                self._gate_id += 1
           
    def cx_all_neighbors(self):
        for i in range(self._num_of_qubits):
            control = i
            target = i+1
            
            # Handles last condition
            if control == self._num_of_qubits - 1:
                target = 0
                
            self._qc.cx(control, target)
            
    def cx_almost_all_neighbors(self):
        for i in range(self._num_of_qubits-1):
            control = i
            target = i+1                
            self._qc.cx(control, target)
        
    def cx_one_to_all(self):       
        index_list = [i for i in range(self._num_of_qubits)]
        
        for j in range(self._num_of_qubits):
            control = j
            temp_index_list = index_list.copy()
            temp_index_list.remove(control)
            for target in temp_index_list:
                self._qc.cx(control, target)
                
    def crx_all_neighbors(self, params=None):
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        
        for i in range(self._num_of_qubits):
            control = i
            target = i+1
            if control == self._num_of_qubits - 1:
                target = 0
            self._qc.crx(params[i], control, target)
            self._all_gate_params['w'+str(self._gate_id)] = params[i]
            self._gate_id += 1
            
    def crx_almost_all_neighbors(self, params=None):
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        
        for i in range(self._num_of_qubits - 1):
            control = i
            target = i+1
            self._qc.crx(params[i], control, target)
            self._all_gate_params['w'+str(self._gate_id)] = params[i]
            self._gate_id += 1
            
    def crx_one_to_all(self, params=None):
        index_list = [i for i in range(self._num_of_qubits)]
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits * (self._num_of_qubits-1))]
        
        for j in range(self._num_of_qubits):
            control = j
            temp_index_list = index_list.copy()
            temp_index_list.remove(control)
            for target in temp_index_list:
                self._qc.crx(params[j+target], control, target)
                self._all_gate_params['w'+str(self._gate_id)] = params[j+target]
                self._gate_id += 1
    
    def crz_all_neighbors(self, params=None):
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        
        for i in range(self._num_of_qubits):
            control = i
            target = i+1
            if control == self._num_of_qubits - 1:
                target = 0
            self._qc.crz(params[i], control, target)
            self._all_gate_params['w'+str(self._gate_id)] = params[i]
            self._gate_id += 1
            
    def crz_almost_all_neighbors(self, params=None):
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits)]
        
        for i in range(self._num_of_qubits - 1):
            control = i
            target = i+1
            self._qc.crz(params[i], control, target)
            self._all_gate_params['w'+str(self._gate_id)] = params[i]
            self._gate_id += 1
            
    def crz_one_to_all(self, params=None):
        index_list = [i for i in range(self._num_of_qubits)]
        if params == None:
            params = [random.uniform(0, 2*np.pi) for i in range(self._num_of_qubits * (self._num_of_qubits-1))]
        
        for j in range(self._num_of_qubits):
            control = j
            temp_index_list = index_list.copy()
            temp_index_list.remove(control)
            for target in temp_index_list:
                self._qc.crz(params[j+target], control, target)
                self._all_gate_params['w'+str(self._gate_id)] = params[j+target]
                self._gate_id += 1
    