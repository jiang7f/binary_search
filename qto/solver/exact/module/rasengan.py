def driver_component(qc: QuantumCircuit, list_qubits:Iterable, list_ancilla:Iterable, bit_string:str, phase:float, mcx_mode:str='linear'):
    # 把|w>转换成|w>
    from qiskit.circuit.library import MCXGate
    for i, v in enumerate(bit_string):
        if v == 0:
            qc.x(list_qubits[i])
    qc.append(MCXGate(len(bit_string)), qargs=list_qubits + [list_ancilla[0]])
    for i, v in enumerate(bit_string):
        if v == 0:
            qc.x(list_qubits[i])
    for i in list_qubits:
        qc.cx(list_ancilla[0], i)
    qc.reset(list_ancilla[0])

    apply_convert(qc, list_qubits, bit_string)
    decompose_phase_gate(qc, list_qubits, list_ancilla, -phase, mcx_mode)
    qc.x(list_qubits[-1])
    decompose_phase_gate(qc, list_qubits, list_ancilla, phase, mcx_mode)
    # 给|w>加相位 pi / 2

    qc.x(list_qubits[-1])
    apply_reverse(qc, list_qubits, bit_string)

    for i, v in enumerate(bit_string):
        if v == 0:
            qc.x(list_qubits[i])
    decompose_phase_gate(qc, list_qubits, list_ancilla, np.pi / 2, mcx_mode)
    for i, v in enumerate(bit_string):
        if v == 0:
            qc.x(list_qubits[i])

# -------