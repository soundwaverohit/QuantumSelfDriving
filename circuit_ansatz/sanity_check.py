## Imports
from gate_configurations import circuit

## Construct a quantum circuit with # of qubits
qc = circuit(4)

## Construct a Walsh-Hadamard Transform (H-cascade)
qc.build_cascade_hadamard()

## Construct a RX-Gate Cascade
#qc.build_cascade_rx()

## Construct a RZ-Gate Cascade
#qc.build_cascade_rz()

## Construct an all-neighbors configuration of CNOT gates
#qc.cx_all_neighbors()

## Construct an one-to-all configuration of CNOT gates
#qc.cx_one_to_all()

## Construct an all-neighbors configuration of Controlled-RX gates
#qc.crx_all_neighbors()

## Construct an one-to-all configuration of Controlled-RX gates
#qc.crx_one_to_all()

## Draw the quantum circuit
print(qc.get_quantum_circuit().draw())

## Print all circuit rotation parameters
print(qc.get_rotation_params())