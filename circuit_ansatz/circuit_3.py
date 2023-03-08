from gate_configurations import circuit

circ_3 = circuit(4)
circ_3.build_cascade_rx()
circ_3.build_cascade_rz()
circ_3.crz_almost_all_neighbors()

