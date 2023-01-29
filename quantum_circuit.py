import pennylane as qml



class VQA(object):
    trained_rotations = None

    def __init__(self, trained_rotations=None):
        self.trained_rotations = trained_rotations

    def create_custom_v1_circuit(self, trainable=True):
        n_qubits = 6
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (16)}
        keras_layer = None

        
        def qml_circuit(inputs, weights):
            """
            Example
             0: ──RX(1.1)───RX(-0.224)──RY(0.37)────╭C───RX(0.37)───RY(-0.253)─────────────────────────────────────────────╭X──╭C──┤
             1: ──RX(3.01)──RX(0.209)───RY(0.354)───╰X──╭C──────────RX(0.354)────RY(-0.3)──────────────────────────────╭X──╰C──│───┤
             2: ──RX(0)─────RX(-0.249)──RY(-0.147)──────╰X─────────╭C────────────RX(-0.147)──RY(-0.435)────────────╭X──╰C──────│───┤
             3: ──RX(1.51)──RX(-0.334)──RY(-0.383)─────────────────╰X───────────╭C───────────RX(-0.383)──RY(0.05)──╰C──────────│───┤
             4: ────────────────────────────────────────────────────────────────╰X─────────────────────────────────────────────│───┤ ⟨Z⟩
             5: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╰X──┤ ⟨Z⟩
            """
            rotatations = weights
            if not trainable:
                rotatations = trained_rotations
            layer_size = 4
            qml.templates.AngleEmbedding(inputs, wires=range(layer_size))

            #first rotations
            for index in range(layer_size):
                qml.RX(rotatations[index], wires=index)
                qml.RY(rotatations[index+4], wires=index)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

            #first output
            qml.CNOT(wires=[3, 4])

            # second rotations
            for index in range(layer_size):
                real_index = layer_size + index + 4
                qml.RX(rotatations[real_index], wires=index)
                qml.RY(rotatations[real_index + 4], wires=index)
            qml.CNOT(wires=[3, 2])
            qml.CNOT(wires=[2, 1])
            qml.CNOT(wires=[1, 0])

            #second output
            qml.CNOT(wires=[0, 5])

            return_qubits = [qml.expval(qml.PauliZ(wires=4)), qml.expval(qml.PauliZ(wires=5))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer

    def create_two_layer_circuit(self, trainable=True):
        n_qubits = 4
        n_layers = 2
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (n_layers, n_qubits)}
        keras_layer = None
        if trainable:
            @qml.qnode(dev)
            def qml_circuit(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

            keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=n_qubits)
        else:
            # example
            #trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]

            if not trained_rotations or len(trained_rotations) != 8:
                raise Exception("Parameter trained_rotations is wrong!")

            @qml.qnode(dev)
            def qml_circuit(inputs, weights):
                """
                Example
                 0: ──RX(0)───────RX(0.406)───╭C───────────────────────────────╭X──RX(-0.171)───╭C──────────╭X──┤ ⟨Z⟩
                 1: ──RX(0)───────RX(0.79)────╰X──╭C───RX(-0.665)──────────────│────────────────╰X──╭C──────│───┤ ⟨Z⟩
                 2: ──RX(0.0475)──RX(-0.499)──────╰X──╭C───────────RX(0.0367)──│────────────────────╰X──╭C──│───┤ ⟨Z⟩
                 3: ──RX(0)───────RX(0.671)───────────╰X───────────────────────╰C──RX(-0.0497)──────────╰X──╰C──┤ ⟨Z⟩
                """
                # input
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.RX(trained_rotations[0], wires=0)
                qml.RX(trained_rotations[1], wires=1)
                qml.RX(trained_rotations[2], wires=2)
                qml.RX(trained_rotations[3], wires=3)
                qml.CNOT(wires=[0,1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.RX(trained_rotations[4], wires=1)
                qml.RX(trained_rotations[5], wires=2)
                qml.CNOT(wires=[3,0])
                qml.RX(trained_rotations[6], wires=0)
                qml.RX(trained_rotations[7], wires=3)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

            keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=n_qubits)

        return keras_layer

    