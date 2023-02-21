from circuit_6 import circ_6

print(circ_6.get_quantum_circuit().draw())

print(len(circ_6.get_rotation_params().keys()), 'gates')