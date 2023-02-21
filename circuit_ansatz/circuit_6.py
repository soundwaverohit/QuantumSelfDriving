from gate_configurations import circuit

circ_6 = circuit(4)
circ_6.build_cascade_rx()
circ_6.build_cascade_rz()
circ_6.crx_one_to_all()
circ_6.build_cascade_rx()
circ_6.build_cascade_rz()