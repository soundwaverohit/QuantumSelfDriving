from gate_configs import circuit

qc = circuit(4)
qc.build_cascade_hadamard()
#qc.build_cascade_rx()
#qc.build_cascade_rz()
qc.cx_all_neighbors()
print(qc._qc.draw())
print(qc.all_gate_params)