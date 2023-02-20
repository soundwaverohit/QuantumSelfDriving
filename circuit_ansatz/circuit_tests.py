from gate_configs import circuit

qc = circuit(4)

#qc.build_cascade_hadamard()
#qc.build_cascade_rx()
#qc.build_cascade_rz()
#qc.cx_all_neighbors()
#qc.cx_one_to_all()
#qc.crx_all_neighbors()
#qc.crx_one_to_all()

print(qc._qc.draw())
print(qc.all_gate_params)