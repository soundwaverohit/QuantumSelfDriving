from circuit_1 import circ_1
import pennylane as qml
from pennylane_qiskit import *

circ = circ_1.get_quantum_circuit()
print(circ.draw())

pnyln_circ = pq.load(circ)

print(qml.draw(pnyln_circ))