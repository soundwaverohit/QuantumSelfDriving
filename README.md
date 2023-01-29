# QuantumSelfDriving

## Project Overview
Implementing a Variatonal Quantum Circuit along with a Machine Learning Approach for a simple steering angle correction model for self driving cars

## Project Approach
1. Build a Variatonal Quantum Circut with 4 qubits
2. Take a set of images of cars driving on the road and label them with an appropriate steering angles for the car.
3. Train a classical CNN model with multiple layers to predict the car's steering angle based on images
4. Interchange the weights to a new model architecture where the dense layer changes with the Variatonal Quantum Circuit
5. Train the quantum circuits with correct steering angles to keep the car on the road
6. Build a simulation to show the results


### Sample Variational Quantum Circuit

Example 1: 6 Qubits Example circuit
0: ──RX(1.1)───RX(-0.224)──RY(0.37)────╭C───RX(0.37)───RY(-0.253)─────────────────────────────────────────────╭X──╭C──┤
1: ──RX(3.01)──RX(0.209)───RY(0.354)───╰X──╭C──────────RX(0.354)────RY(-0.3)──────────────────────────────╭X──╰C──│───┤
2: ──RX(0)─────RX(-0.249)──RY(-0.147)──────╰X─────────╭C────────────RX(-0.147)──RY(-0.435)────────────╭X──╰C──────│───┤
3: ──RX(1.51)──RX(-0.334)──RY(-0.383)─────────────────╰X───────────╭C───────────RX(-0.383)──RY(0.05)──╰C──────────│───┤
4: ────────────────────────────────────────────────────────────────╰X─────────────────────────────────────────────│───┤ ⟨Z⟩
5: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╰X──┤ ⟨Z⟩


Example 2: 4 Qubits Example circuit
0: ──RX(0)───────RX(0.406)───╭C───────────────────────────────╭X──RX(-0.171)───╭C──────────╭X──┤ ⟨Z⟩
1: ──RX(0)───────RX(0.79)────╰X──╭C───RX(-0.665)──────────────│────────────────╰X──╭C──────│───┤ ⟨Z⟩
2: ──RX(0.0475)──RX(-0.499)──────╰X──╭C───────────RX(0.0367)──│────────────────────╰X──╭C──│───┤ ⟨Z⟩
3: ──RX(0)───────RX(0.671)───────────╰X───────────────────────╰C──RX(-0.0497)──────────╰X──╰C──┤ ⟨Z⟩
                

Both models spitting out steering angles along with an optimizer to be implemented