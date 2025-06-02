import numpy as np
from qiskit import QuantumCircuit
from typing import Iterable

from qto.provider import Provider
from qto.utils import iprint
from qto.model import ModelOption
from .abstract import ExactCircuit, ExactSolver
from .option import QbsCircuitOption

# 用更多比特，对应cxmode为linear，拓扑可能更差
def mcx_n_anc_log_decompose(circuit: QuantumCircuit, control_qubits, target_qubit, ancillary_qubits):
    """
    This function implements the multi-controlled-X gate using the Toffoli gate.
    """
    if len(control_qubits) == 0:
        circuit.x(target_qubit)
    elif len(control_qubits) == 1:
        circuit.cx(control_qubits[0], target_qubit)
        return 0
    elif len(control_qubits) == 2:
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        return 1
    else:
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        res = mcx_n_anc_log_decompose(
            circuit,
            control_qubits[2:] + [ancillary_qubits[0]],
            target_qubit,
            ancillary_qubits[1:],
        )
        circuit.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[0])
        return res
    
def mcx_gate_decompose(qc: QuantumCircuit, list_controls:Iterable, qubit_target:int, list_ancilla:Iterable, mcx_mode):
    if mcx_mode == 'constant':
        # 自动分解，34 * 非零元
        qc.mcx(list_controls, qubit_target, list_ancilla[0], mode='recursion')
    elif mcx_mode == 'linear':
        # log 但是用更多比特，映射后可能反而更差
        mcx_n_anc_log_decompose(qc,list_controls, qubit_target, list_ancilla)
    else:
        qc.mcx(list_controls, qubit_target, list_ancilla[0])


def decompose_phase_gate(qc: QuantumCircuit, list_qubits:list, list_ancilla:list, phase:float, mcx_mode) -> QuantumCircuit:
    """
    Decompose a phase gate into a series of controlled-phase gates.
    Args:
        qc
        list_qubits
        list_ancilla
        phase (float): the phase angle of the phase gate.
        mcx_mode (str): the type of ancillary qubits used in the controlled-phase gates.
            'constant': use a constant number of ancillary qubits for all controlled-phase gates.
            'linear': use a linear number of ancillary qubits to guarantee logarithmic depth.
    Returns:
        QuantumCircuit: the circuit that implements the decomposed phase gate.
    """
    num_qubits = len(list_qubits)
    if num_qubits == 1:
        qc.p(phase, list_qubits[0])
    elif num_qubits == 2:
        qc.cp(phase, list_qubits[0], list_qubits[1])
    else:
        # convert into the multi-cx gate 
        # partition qubits into two sets
        half_num_qubit = num_qubits // 2
        qr1 = list_qubits[:half_num_qubit]
        qr2 = list_qubits[half_num_qubit:]
        qc.rz(-phase/2, list_ancilla[0])
        # use ", mode='recursion'" without transpile will raise error 'unknown instruction: mcx_recursive'
        mcx_gate_decompose(qc, qr1, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr2, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(-phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr1, list_ancilla[0], list_ancilla[1:], mcx_mode)
        qc.rz(phase/2, list_ancilla[0])
        mcx_gate_decompose(qc, qr2, list_ancilla[0], list_ancilla[1:], mcx_mode)


def apply_convert(qc: QuantumCircuit, list_qubits, bit_string):
    num_qubits = len(bit_string)
    for i in range(0, num_qubits - 1):
        qc.cx(list_qubits[i + 1], list_qubits[i])
        if bit_string[i] == bit_string[i + 1]:
            qc.x(list_qubits[i])
    qc.h(list_qubits[num_qubits - 1])
    qc.x(list_qubits[num_qubits - 1])

def apply_reverse(qc: QuantumCircuit, list_qubits, bit_string):
    num_qubits = len(bit_string)
    qc.x(list_qubits[num_qubits - 1])
    qc.h(list_qubits[num_qubits - 1])
    for i in range(num_qubits - 2, -1, -1):
        if bit_string[i] == bit_string[i + 1]:
            qc.x(list_qubits[i])
        qc.cx(list_qubits[i + 1], list_qubits[i])
        
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

def new_compnt(qc: QuantumCircuit, Hd_bitstr_list, anc_idx, mcx_mode):
    for idx, hdi_vct in enumerate(Hd_bitstr_list):
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        # qc.h(hdi_bitstr[0])
        driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, np.pi/4, mcx_mode)
        # qc.h(hdi_bitstr[0])

class QbsCircuit(ExactCircuit[QbsCircuitOption]):
    def __init__(self, circuit_option: QbsCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()

    def inference(self):
        final_qc = self.inference_circuit
        counts = self.circuit_option.provider.get_counts_with_time(final_qc, shots=self.circuit_option.shots)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs
    
    def create_circuit(self) -> QuantumCircuit:
        num_qubits = self.model_option.num_qubits        
        mcx_mode = self.circuit_option.mcx_mode

        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))

        qc = self.circuit_option.provider.transpile(qc)
        for i in np.nonzero(self.model_option.feasible_state)[0]:
            qc.x(i)
        
        Hd_bitstr_list = np.tile(self.model_option.Hd_bitstr_list, (3, 1))
        new_compnt(
            qc,
            Hd_bitstr_list,
            anc_idx,
            mcx_mode,
        )
        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc
        pass

class QbsSolver(ExactSolver):
    def __init__(
        self,
        *,
        prb_model,
        provider: Provider,
        shots: int = 1024,
        mcx_mode: str = "constant",
        eps: float = 1e-0
    ):
        super().__init__(prb_model)
        
        self.circuit_option = QbsCircuitOption(
            provider=provider,
            shots=shots,
            mcx_mode=mcx_mode,
        )
        self.eps = eps

    @property
    def circuit(self) -> QbsCircuit:
        if self._circuit is None:
            self._circuit = QbsCircuit(self.circuit_option, self.model_option)
        return self._circuit

    def check(self, mid):
        
        ''' 振幅放大后 有没有b '''
        if mid > 530:
            return True
        else:
            return False
        

    def _solve_impl(self):
        # 估计取值范围
        pass
        bound_left = 0
        bound_right = 10000
        print(self.circuit.inference())
        iprint("Starting binary search...")
        # binary search
        while bound_right - bound_left > self.eps:
            mid = (bound_left + bound_right) / 2
            iprint(f"Check: {mid}")
            if self.check(mid):
                bound_right = mid  # 缩小右边界，逼近最小满足条件的值
            else:
                bound_left = mid   # 舍弃当前和左边，逼近满足条件的区间
        
        # 得到最后一个b
        iprint(f"Optimal value found: {bound_left}")