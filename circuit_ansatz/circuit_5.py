from gate_configurations import circuit

circ_5 = circuit(4)
circ_5.build_cascade_rx()
circ_5.build_cascade_rz()
circ_5.crz_one_to_all()
circ_5.build_cascade_rx()
circ_5.build_cascade_rz()