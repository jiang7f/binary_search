import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qto.solver.vqa.abstract.vqa_solver import VqaSolver
from qto.solver.vqa.option import LayerCircuitOption, OptimizerOption
from qto.model import ModelOption
from qto.model import LinearConstrainedBinaryOptimization as LcboModel

from .abstract import VqaCircuit, Optimizer
from ...provider import Provider
from .circuit_decompose.circuit_components import obj_compnt, commute_compnt


class HeaCircuit(VqaCircuit[LayerCircuitOption]):
    def __init__(self, circuit_option: LayerCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()

    def get_num_params(self):
        return self.circuit_option.num_layers * self.model_option.num_qubits * 3
    
    def inference(self, params):
        final_qc = self.inference_circuit.assign_parameters(params)
        counts = self.circuit_option.provider.get_counts_with_time(final_qc, shots=self.circuit_option.shots)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs

    def create_circuit(self) -> QuantumCircuit:
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits

        qc = QuantumCircuit(num_qubits, num_qubits)
        r1_params = [[Parameter(f'r1_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
        r2_params = [[Parameter(f'r2_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
        r3_params = [[Parameter(f'r3_params[{i}_{j}]') for j in range(num_qubits)] for i in range(num_layers)]
        for layer in range(num_layers):
            for i in range(num_qubits):
                qc.rz(r1_params[layer][i], i)
                qc.ry(r2_params[layer][i], i)
                qc.rz(r3_params[layer][i], i)
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc
    
class HeaSolver(VqaSolver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int,
        shots: int = 1024,
    ):
        super().__init__(prb_model, optimizer)
        self.circuit_option = LayerCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = HeaCircuit(self.circuit_option, self.model_option)
        return self._circuit