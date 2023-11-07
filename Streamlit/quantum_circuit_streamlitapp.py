import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import circuit_drawer, plot_state_city
import matplotlib.pyplot as plt
from streamlit_ws_localstorage import injectWebsocketCode, getOrCreateUID

conn = injectWebsocketCode(hostPort='linode.liquidco.in', uid=getOrCreateUID())

def update(circuit, circuit_visualization, **n):
    opr = st.session_state.gate_operations.copy()

    while len(opr) > st.session_state._qubit:
            opr.pop()
            
    for qubit, gate_ops in enumerate(opr):
        for gate in gate_ops:
            if gate == "Hadamard":
                circuit.h(qubit)
            elif gate == "Pauli-X":
                circuit.x(qubit)
            elif gate == "Pauli-Y":
                circuit.y(qubit)
            elif gate == "Pauli-Z":
                circuit.z(qubit)
            elif gate == "CNOT":
                control = (qubit + 1) % st.session_state._qubit
                circuit.cx(qubit, control)

    circuit_visualization.image(update_circuit_visualization(circuit), use_column_width=True)

def update_circuit_visualization(circuit):
    circuit_drawer_file = "circuit_drawer.png"

    circuit_drawer(circuit, output='mpl', filename=circuit_drawer_file)
    image = plt.imread(circuit_drawer_file)
    plt.close()  # Close the figure to avoid overlapping images
    return image


def main():
    global conn
    if "state_number" not in st.session_state:
        st.session_state.state_number = 0
    st.title("Quantum Composer")
    
    # Sidebar options
    num_qubits = st.sidebar.number_input("Number of Qubits", min_value=1, max_value=5, value=2, step=1)


    gate_operations = [[] for _ in range(num_qubits)]
    # Initialize session-states
    if "_qubit" not in st.session_state:
        st.session_state._qubit = num_qubits

    if st.session_state._qubit != num_qubits:
        st.session_state._qubit = num_qubits;
        if len(st.session_state.gate_operations) < st.session_state._qubit:
            # st.session_state.gate_operations.append([])
            st.session_state.gate_operations.append([])

    if "gate_operations" not in st.session_state:
        st.session_state.gate_operations = gate_operations

        
    # Initialize a Quantum Circuit
    circuit = QuantumCircuit(st.session_state._qubit)
    
    # Quantum Gates
    gate_options = ["Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z", "CNOT"]

    # Display the Quantum Circuit
    st.subheader("Quantum Circuit")
    circuit_visualization = st.empty()

    # Add Gates to the Circuit
    st.sidebar.subheader("Add Gates")

    # element= True

    if st.sidebar.button("Save", key="s"):

        r = st.session_state.gate_operations.copy()
        # for i in range()
        print(r)
        for i in r:
            i.append(1)
        print(r)
        conn.setLocalStorageVal(key=st.session_state.state_number, val=r)
        for i in r:
            i.remove(1)
        del r

        st.session_state.state_number += 1
    for qubit in range(num_qubits):
        gate_label = f"Gate - Qubit {qubit}"
        selected_gate = st.sidebar.selectbox(gate_label, gate_options, key=f"gate-{qubit}")
        selected_qubit = qubit


        if st.sidebar.button(f"Apply Gate {qubit}", key=f"apply-{qubit}"):
            
            st.session_state.gate_operations[selected_qubit].append(selected_gate)
            
            update(circuit, circuit_visualization)            


    if st.session_state.state_number:
        for i in range(st.session_state.state_number):
            if st.sidebar.button(f"state_${i + 1}", key=i):
                vt= conn.getLocalStorageVal(key=i)
                parsed_arr_1 = vt.split(",")
                r = [[]]

                for t in parsed_arr_1:
                    if t.isdigit():
                        r.append([])
                    else:
                        r[-1].append(t)
            


                st.session_state.gate_operations = r
                update(circuit, circuit_visualization)
    # Execute the Circuit
    st.sidebar.subheader("Execute Circuit")
    backend_options = ["qasm_simulator", "statevector_simulator", "unitary_simulator"]
    selected_backend = st.sidebar.selectbox("Select Backend", backend_options)

    if st.sidebar.button("Run Circuit"):
        # Reset the circuit to remove the previously applied gates
        circuit = QuantumCircuit(num_qubits)        

        # Update the circuit with the selected gates
        for qubit, gate_ops in enumerate(st.session_state.gate_operations):
            for gate in gate_ops:
                if gate == "Hadamard":
                    circuit.h(qubit)
                elif gate == "Pauli-X":
                    circuit.x(qubit)
                elif gate == "Pauli-Y":
                    circuit.y(qubit)
                elif gate == "Pauli-Z":
                    circuit.z(qubit)
                elif gate == "CNOT":
                    control = (qubit + 1) % num_qubits
                    circuit.cx(qubit, control)

        backend = Aer.get_backend(selected_backend)
        job = execute(circuit, backend)
        result = job.result()

        if selected_backend == "qasm_simulator":
            counts = result.get_counts()
            st.subheader("Measurement Results")
            st.text(counts)
        elif selected_backend == "statevector_simulator":
            statevector = result.get_statevector()
            st.subheader("Final Statevector")
            st.text(statevector)
            st.subheader("Statevector Visualization")
            st.pyplot(plot_state_city(statevector))
        elif selected_backend == "unitary_simulator":
            unitary = result.get_unitary()
            st.subheader("Final Unitary")
            st.text(unitary)


if __name__ == "__main__":
    main()
