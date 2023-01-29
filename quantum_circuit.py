import pennylane as qml



class MyVQAClass(object):
    trained_rotations = None

    def __init__(self, trained_rotations=None):
        self.trained_rotations = trained_rotations

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

    def create_custom_v1_circuit(self, trainable=True):
        n_qubits = 6
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (16)}
        keras_layer = None

        @qml.qnode(dev)
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

    def create_custom_v2_circuit(self, trainable=True):
        n_qubits = 6
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (16)}
        keras_layer = None

        @qml.qnode(dev)
        def qml_circuit(inputs, weights):
            """
            Example
             0: ──RX(0)────RX(-0.169)──RY(-0.214)──╭C───RX(-0.214)───RY(0.227)───────────────╭C─────────────────────────────────┤
             1: ──RX(0)────RX(-0.232)──RY(0.324)───╰X──╭C────────────RX(0.324)───RY(0.344)───╰X───────────╭C────────────────────┤
             2: ──RX(0)────RX(0.0535)──RY(-0.177)──────╰X───────────╭C───────────RX(-0.177)───RY(-0.426)──╰X────────────╭C──────┤
             3: ──RX(3.2)──RX(-0.27)───RY(-0.37)────────────────────╰X──────────╭C────────────RX(-0.37)────RY(-0.0912)──╰X──╭C──┤
             4: ────────────────────────────────────────────────────────────────╰X──────────────────────────────────────────│───┤ ⟨Z⟩
             5: ────────────────────────────────────────────────────────────────────────────────────────────────────────────╰X──┤ ⟨Z⟩
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
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

            #second output
            qml.CNOT(wires=[3, 5])

            return_qubits = [qml.expval(qml.PauliZ(wires=4)), qml.expval(qml.PauliZ(wires=5))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer

    def create_custom_v3_circuit(self, trainable=True):
        n_qubits = 4
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (16)}

        @qml.qnode(dev)
        def qml_circuit(inputs, weights):
            """
            Example
            0: ──RX(0.0119)──RX(0.244)────RY(-0.327)──╭C──RX(-0.327)──RY(0.422)───────╭X──╭C──┤
            1: ──RX(0)───────RX(-0.0765)──RY(0.366)───╰X──RX(0.366)───RY(-0.185)──╭C──│───╰X──┤ ⟨Z⟩
            2: ──RX(0)───────RX(-0.0616)──RY(-0.246)──╭X──RX(-0.246)──RY(0.174)───│───╰C──╭X──┤ ⟨Z⟩
            3: ──RX(0.246)───RX(0.148)────RY(0.315)───╰C──RX(0.315)───RY(-0.12)───╰X──────╰C──┤
            """
            rotatations = weights
            if not trainable:
                rotatations = trained_rotations
            layer_size = 4
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            #first part
            for index in range(layer_size):
                qml.RX(rotatations[index], wires=index)
                qml.RY(rotatations[index+4], wires=index)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 2])

            #second part
            for index in range(layer_size):
                real_index = layer_size + index + 4
                qml.RX(rotatations[real_index], wires=index)
                qml.RY(rotatations[real_index+4], wires=index)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 0])

            #generate output
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 2])

            return_qubits = [qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliZ(wires=2))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer

    def create_custom_v4_circuit(self, trainable=True):
        n_qubits = 4
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (24)}

        @qml.qnode(dev)
        def qml_circuit(inputs, weights):
            """
            Example
             0: ──RX(3.81)──RX(0.0541)──RY(-0.138)──╭C──RX(-0.138)──RY(-0.274)──────╭X──RX(-0.274)──RY(0.149)───╭C──┤
             1: ──RX(1.59)──RX(-0.23)───RY(0.285)───╰X──RX(0.285)───RY(0.0388)──╭C──│───RX(0.0388)──RY(-0.117)──╰X──┤ ⟨Z⟩
             2: ──RX(1.43)──RX(0.174)───RY(-0.253)──╭X──RX(-0.253)──RY(0.171)───│───╰C──RX(0.171)───RY(0.209)───╭X──┤ ⟨Z⟩
             3: ──RX(1.53)──RX(-0.146)──RY(-0.32)───╰C──RX(-0.32)───RY(-0.015)──╰X──────RX(-0.015)──RY(-0.289)──╰C──┤
            """
            rotatations = weights
            if not trainable:
                rotatations = trained_rotations
            layer_size = 4
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))


            #first part
            for index in range(layer_size):
                qml.RX(rotatations[index], wires=index)
                qml.RY(rotatations[index+4], wires=index)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 2])

            #second part
            for index in range(layer_size):
                real_index = layer_size + index + 4
                qml.RX(rotatations[real_index], wires=index)
                qml.RY(rotatations[real_index+4], wires=index)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 0])

            # last part
            for index in range(layer_size):
                real_index = (layer_size*2) + index + (4*2)
                qml.RX(rotatations[real_index], wires=index)
                qml.RY(rotatations[real_index + 4], wires=index)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 2])

            return_qubits = [qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliZ(wires=2))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer

    def create_custom_v5_circuit(self, trainable=True):
        n_qubits = 4
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (8)}

        @qml.qnode(dev)
        def qml_circuit(inputs, weights):
            """
            Example
             0: ──RX(0.0508)──RX(0.117)───RY(-0.455)──╭C──┤
             1: ──RX(0)───────RX(-0.5)────RY(0.313)───╰X──┤ ⟨Z⟩
             2: ──RX(0)───────RX(-0.273)──RY(-0.286)──╭X──┤ ⟨Z⟩
             3: ──RX(0.0201)──RX(-0.289)──RY(0.118)───╰C──┤
            """
            rotatations = weights
            if not trainable:
                rotatations = trained_rotations
            layer_size = 4
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            #first part
            for index in range(layer_size):
                qml.RX(rotatations[index], wires=index)
                qml.RY(rotatations[index+4], wires=index)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 2])

            return_qubits = [qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliZ(wires=2))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer

    def create_custom_v6_circuit(self, trainable=True):
        n_qubits = 8
        trained_rotations = self.trained_rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (28)}

        @qml.qnode(dev)
        def qml_circuit(inputs, weights):
            """
            Example
             0: ──RX(0.0132)──RX(0.0159)───RY(-0.28)────╭C───────────────────────────────────────────────────────────────────────────────────────┤
             1: ──RX(2.64)────RX(-0.0397)──RY(-0.179)───╰X──RX(0.32)────RY(-0.232)────╭C─────────────────────────────────────────────────────────┤
             2: ──RX(0)───────RX(0.0718)───RY(-0.0534)────────────────────────────────╰X──RX(0.0256)───RY(-0.264)──╭C────────────────────────────┤
             3: ──RX(0)───────RX(0.171)────RY(-0.14)───────────────────────────────────────────────────────────────╰X──RX(0.1)──────RY(-0.164)───┤ ⟨Z⟩
             4: ──RX(0)───────RX(-0.28)────RY(-0.225)──────────────────────────────────────────────────────────────╭X──RX(-0.0135)──RY(-0.0187)──┤ ⟨Z⟩
             5: ──RX(0)───────RX(-0.179)───RY(0.0884)─────────────────────────────────╭X──RX(-0.0633)──RY(0.23)────╰C────────────────────────────┤
             6: ──RX(0)───────RX(-0.0534)──RY(0.0693)───╭X──RX(-0.285)──RY(-0.00816)──╰C─────────────────────────────────────────────────────────┤
             7: ──RX(3.28)────RX(-0.14)────RY(0.0137)───╰C───────────────────────────────────────────────────────────────────────────────────────┤
            """
            rotatations = weights
            if not trainable:
                rotatations = trained_rotations
            layer_size = 8
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            #first part
            for index in range(layer_size):
                qml.RX(rotatations[index], wires=index)
                qml.RY(rotatations[index+4], wires=index)

            #layer 2
            qml.CNOT(wires=[0, 1])
            qml.RX(rotatations[16], wires=1)
            qml.RY(rotatations[17], wires=1)
            qml.CNOT(wires=[7, 6])
            qml.RX(rotatations[18], wires=6)
            qml.RY(rotatations[19], wires=6)

            #layer 3
            qml.CNOT(wires=[1, 2])
            qml.RX(rotatations[20], wires=2)
            qml.RY(rotatations[21], wires=2)
            qml.CNOT(wires=[6, 5])
            qml.RX(rotatations[22], wires=5)
            qml.RY(rotatations[23], wires=5)

            # layer 4
            qml.CNOT(wires=[2, 3])
            qml.RX(rotatations[24], wires=3)
            qml.RY(rotatations[25], wires=3)
            qml.CNOT(wires=[5, 4])
            qml.RX(rotatations[26], wires=4)
            qml.RY(rotatations[27], wires=4)

            return_qubits = [qml.expval(qml.PauliZ(wires=3)), qml.expval(qml.PauliZ(wires=4))]

            return return_qubits

        keras_layer = qml.qnn.KerasLayer(qml_circuit, weight_shapes, output_dim=2)
        return keras_layer