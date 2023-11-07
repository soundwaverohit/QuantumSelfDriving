import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer


def apply_gate(circuit, gate, qubit):
    if gate == 'X':
        circuit.x(qubit)
    elif gate == 'Y':
        circuit.y(qubit)
    elif gate == 'Z':
        circuit.z(qubit)
    elif gate == 'H':
        circuit.h(qubit)
    elif gate == 'S':
        circuit.s(qubit)
    elif gate == 'T':
        circuit.t(qubit)
    elif gate == 'CX':
        control_qubit = st.number_input(
            "Enter the control qubit index:",
            value=0,
            min_value=0,
            max_value=num_qubits - 1,
            key=f"cx_control_{qubit}"
        )
        circuit.cx(control_qubit, qubit)
    elif gate == 'CCX':
        control_qubit1 = st.number_input(
            "Enter the first control qubit index:",
            value=0,
            min_value=0,
            max_value=num_qubits - 1,
            key=f"ccx_control1_{qubit}"
        )
        control_qubit2 = st.number_input(
            "Enter the second control qubit index:",
            value=0,
            min_value=0,
            max_value=num_qubits - 1,
            key=f"ccx_control2_{qubit}"
        )
        circuit.ccx(control_qubit1, control_qubit2, qubit)


def compute_circuit(circuit):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    statevector = result.get_statevector()
    return statevector


st.title("Quantum Composer")

# Ask user for the number of qubits
num_qubits = st.slider("Number of Qubits:", min_value=1, max_value=10, value=1, step=1)

# Create a list to store the gates for each qubit
gate_list = [[] for _ in range(num_qubits)]

# Create the circuit
circuit = QuantumCircuit(num_qubits)

# Function to clear the circuit
def clear_circuit():
    global gate_list
    global circuit
    gate_list = [[] for _ in range(num_qubits)]
    circuit = QuantumCircuit(num_qubits)
    st.write(circuit_drawer(circuit, output='mpl'))
    #return circuit

# Iterate over qubits
for qubit in range(num_qubits):
    st.header(f"Qubit {qubit}")
    gate_form = st.form(key=f"gate_form_{qubit}")
    with gate_form:

        selected_gate = st.selectbox(
            f"Select a gate for Qubit {qubit}:",
            ('None', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CX', 'CCX')
        )
    gate_list[qubit].append(selected_gate)
        
        #if selected_gate != 'None':
        #gate_list[qubit].append(selected_gate)
            #apply_gate(circuit, selected_gate, qubit)
    gate_form.form_submit_button("Add Gate")



variable = True
while variable == True:
    if gate_form.form_submit_button("Add more gates"):
        with gate_form:
            selected_gate1 = st.selectbox(
                f"Select a gate for Qubit {qubit}:",
                ('None', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CX', 'CCX'), key="3q"
            )
        gate_list[qubit].append(selected_gate1)

                
                #if selected_gate != 'None':
                #gate_list[qubit].append(selected_gate)
                    #apply_gate(circuit, selected_gate, qubit)
        gate_form.form_submit_button("Add another Gate")
        

        if gate_form.form_submit_button("Finished adding gates"):
            variable = False
            





       
            #apply_gate(circuit, selected_gate, qubit)



# Apply the gates from the gate list to the circuit
for qubit, gates in enumerate(gate_list):
    for gate in gates:
        apply_gate(circuit, gate, qubit)

# Circuit visualization
st.header("Circuit Visualization")
st.write(circuit_drawer(circuit, output='mpl'))

# Buttons to clear the circuit and compute the circuit
#col1, col2 = st.columns(2)
if st.button("Clear Circuit"):
    clear_circuit()

if st.button("Compute Circuit"):
    statevector = compute_circuit(circuit)
    st.text("Final statevector:")
    st.write(statevector)
