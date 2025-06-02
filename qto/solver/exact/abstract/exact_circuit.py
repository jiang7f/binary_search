from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Generic, TypeVar
from qto.circuit import Circuit, T, CircuitOption
from qto.model import ModelOption
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeKyiv, FakeQuebec, FakeAlmadenV2, FakeBelemV2, FakeSantiagoV2

class ExactCircuit(Circuit[T], ABC):
    def __init__(self, circuit_option: T, model_option: ModelOption):
        super().__init__(circuit_option, model_option)