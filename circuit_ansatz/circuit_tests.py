from circuit_6 import circuit

qc = circuit(4)
qc.build_cascade_hadamard()
qc.build_cascade_rx()
qc.build_cascade_rz()
print(qc._qc.draw())
print(qc.all_gate_params)