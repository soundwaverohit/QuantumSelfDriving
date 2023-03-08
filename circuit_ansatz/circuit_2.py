from gate_configurations import circuit

circ_2 = circuit(4)
circ_2.build_cascade_rx()
circ_2.build_cascade_rz()
circ_2.cx_almost_all_neighbors()
