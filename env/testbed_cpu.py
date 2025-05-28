should_print = True

from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from qto.solver.vqa.optimizer import CobylaOptimizer, AdamOptimizer
from qto.solver.vqa import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver, 
)
from qto.provider import (
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

# model ----------------------------------------------
m = LcboModel()
x = m.addVars(5, name="x")
m.setObjective((x[0] + x[1])* x[3] + x[2], "max")

m.addConstr(x[0] + x[1] - x[2] == 0)
m.addConstr(x[2] + x[3] - x[4] == 1)

print(m.lin_constr_mtx)
print(m)
optimize = m.optimize()
print(f"optimize_cost: {optimize}\n\n")
# sovler ----------------------------------------------
opt = CobylaOptimizer(max_iter=200)
# gpu = AerGpuProvider()
aer = DdsimProvider()
solver = ChocoSolver(
    prb_model=m,  # 问题模型
    optimizer=opt,  # 优化器
    provider=aer,  # 提供器（backend + 配对 pass_mannager ）
    num_layers=1,
    # mcx_mode="linear",
)
print(solver.circuit_analyze(['depth', 'width', 'culled_depth', 'num_one_qubit_gates']))
result = solver.solve()
eval = solver.evaluation()
print(result)
print(eval)
print("Environment configuration is successful!")
