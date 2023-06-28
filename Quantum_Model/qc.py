"""
ROT3:
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        
        # Circuit 2 - Decision-making
        for i in range(4):
            qml.RX(weights[i][0], wires=i)
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])

            #qml.CNOT(wires=[3, 0])
        
        # Circuit 3 - Control
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
        
        
        # Circuit 4 - Additional functionality for self-driving car
        for i in range(4):
            qml.RZ(weights[i][2], wires=i)
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        
ROT 15


trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(weights[0][0], wires=0)
        qml.Rot(inputs[0][0], inputs[0][1], inputs[0][2], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Rot(inputs[1][0], inputs[1][1], inputs[1][2], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[2][2], wires=2)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.RZ(weights[3][2], wires=3)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]



    return circuit(inputs, weights)

    
    ROT 16

    def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(weights[0][0], wires=0)
        qml.Rot(inputs[0][0], inputs[0][1], inputs[0][2], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Rot(inputs[1][0], inputs[1][1], inputs[1][2], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[2][2], wires=2)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.RZ(weights[3][2], wires=3)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3,0])
        #qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=0)
        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]



    return circuit(inputs, weights)
    

ROT 17
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(weights[0][0], wires=0)
        qml.Rot(inputs[0][0], inputs[0][1], inputs[0][2], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Rot(inputs[1][0], inputs[1][1], inputs[1][2], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[2][2], wires=2)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.RZ(weights[3][2], wires=3)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3,0])
        
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=0)
        qml.RX(weights[0][0], wires=3)
        qml.Rot(inputs[0][0], inputs[0][1], inputs[0][2], wires=3)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3,0])
        qml.RZ(weights[2][2], wires=2)
        qml.Rot(inputs[3][0], inputs[3][1], inputs[3][2], wires=3)
        qml.RZ(weights[3][2], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.Rot(inputs[1][0], inputs[1][1], inputs[1][2], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    



    return circuit(inputs, weights)


ROT 18

def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(weights[0][0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2][2], wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[3][2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    



    return circuit(inputs, weights)


    
ROT 19
trained_rotations = [0.294, 0.212, 0.0129, 0.778, 0.773, -0.386, 0.77, -0.0472]
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(inputs[0][0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(inputs[2][2], wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[3][2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    



    return circuit(inputs, weights)

    
ROT 20
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)

    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        qml.RX(inputs[0][0], wires=0)
        qml.RX(weights[0][0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(inputs[2][2], wires=1)
        qml.RY(weights[0][0], wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[3][2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    



    return circuit(inputs, weights)


    ROT 21

def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        
        # Circuit 2 - Decision-making
        for i in range(4):
            qml.RX(inputs[i][0], wires=i)
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])

            #qml.CNOT(wires=[3, 0])
        
        # Circuit 3 - Control
        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
        
        
        # Circuit 4 - Additional functionality for self-driving car
        for i in range(4):
            qml.RZ(weights[i][2], wires=i)
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires= [3,0])

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


    
Circuit 23
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))

        for i in range(4):
            qml.RX(inputs[i][0], wires=i)
            qml.RX(weights[i][0], wires=i)

        qml.CNOT(wires=[0,1])

        for i in range(4):
            qml.RY(inputs[i][0], wires=i)
            qml.RY(weights[i][0], wires=i)

        qml.CNOT(wires=[1,2])

        for i in range(4):
            qml.RZ(inputs[i][0], wires=i)
            qml.RZ(weights[i][0], wires=i)

        qml.CNOT(wires=[2,3])
        qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
        qml.CNOT(wires=[3,0])


        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)

Circuit 24
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))
        wires=[0, 1, 2,3]

        for i in range(4):
            qml.Rot(inputs[i][0], inputs[i][1], inputs[i][2], wires=i)
            qml.CNOT(wires=[wires[i], wires[(i + 1) % 4]])


        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)

Circuit 25:
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))
        wires=[0, 1, 2,3]

        qml.Rot(inputs[0][0], inputs[1][1], inputs[2][2], wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.Rot(inputs[3][0], inputs[4][1], inputs[5][2], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.Rot(inputs[6][0], inputs[7][1], inputs[8][2], wires=wires[2])


        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)


Circuit 26:
def variational_quantum_circuit(inputs, weights):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        #qml.templates.AngleEmbedding(inputs, wires=range(4))
        wires=[0, 1, 2,3]

        for layer in range(4):
            for qubit in range(4):
                qml.Rot(inputs[layer * 4 + qubit][0],
                     inputs[layer * 4 + qubit][1],
                     inputs[layer * 4 + qubit][2], wires=wires[qubit])

            for qubit in range(4 - 1):
                qml.CNOT(wires=[wires[qubit], wires[qubit + 1]])


        
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return circuit(inputs, weights)



"""