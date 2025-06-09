should_print = True

from qto.model import LinearConstrainedBinaryOptimization as LcboModel, set_coeff_type
# from qto.solver.vqa.optimizer import CobylaOptimizer, AdamOptimizer
from qto.provider import (
     AerProvider, AerGpuProvider, DdsimProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, 
)
from qto.solver.exact import (
    QbsSolver,
)
set_coeff_type(float)
# model ----------------------------------------------
m = LcboModel() 
x = m.addVars(3, name="x")
b = 1
m.setObjective(0.312 * x[0] + 0.15 * x[1] + x[2] * x[1] - b, "min")
# m.addConstr(x[0] + x[1] + x[2] == 2)
# m.addConstr(x[0] + x[1] == 1)
# exit()
m.addConstr(x[0] + x[1] - x[2] == 0)
# m.addConstr(x[2] + x[3] - x[4] == 1)

print(m.lin_constr_mtx)
# exit()
# m.set_
# penalty_lambda(0)
print(m)
optimize = m.optimize()
print(f"optimize_cost: {optimize}\n\n")

# sovler ----------------------------------------------
aer = DdsimProvider()
gpu = AerGpuProvider()
fake = FakeBrisbaneProvider()
# opt = AdamOptimizer(max_iter=200)
solver = QbsSolver(
    prb_model=m,  # 问题模型
    provider=aer,  # 提供器（backend + 配对 pass_mannager
    shots=1024, 
    # mcx_mode="linear",
)
# print(solver.circuit_analyze(['depth', 'width', 'culled_depth', 'num_one_qubit_gates']))
result = solver.solve()
# eval = solver.evaluation()
# print(eval)
# print(opt.cost_history)
