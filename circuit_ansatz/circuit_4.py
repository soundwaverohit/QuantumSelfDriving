from gate_configurations import circuit

circ_4 = circuit(4)
circ_4.build_cascade_rx()
circ_4.build_cascade_rz()
circ_4.crx_almost_all_neighbors()

