import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from itertools import product
import mip
from typing import Dict, Any
import restore_vars


def load_problem(problem_file: str):
    """Parse a problem file and return numpy arrays.
    Args:
        problem_file: file describing the problem

    Returns: A bunch of stuff.
    ...
    """
    with open(problem_file, "r") as input_data_file:
        input_data = input_data_file.read()
    lines = input_data.split("\n")
    parts = lines[0].split()
    nfac = int(parts[0])
    ncus = int(parts[1])
    print(f"{nfac} Facilities and {ncus} Customers ({nfac * ncus:,} in product)")

    # Facility and customer locations, and the capacities and demands.
    f_locs = np.zeros((nfac, 2), np.float64)
    c_locs = np.zeros((ncus, 2), np.float64)
    f_caps = np.zeros(nfac, np.int32)
    c_dems = np.zeros(ncus, np.int32)
    f_costs = np.zeros(nfac, np.float64)

    for i, line_no in enumerate(range(1, nfac + 1)):
        parts = lines[line_no].split()
        f_costs[i] = float(parts[0])
        f_caps[i] = int(parts[1])
        f_locs[i] = [float(parts[2]), float(parts[3])]

    for i, line_no in enumerate(range(nfac + 1, nfac + 1 + ncus)):
        parts = lines[line_no].split()
        c_dems[i] = int(parts[0])
        c_locs[i] = [float(parts[1]), float(parts[2])]

    dists = cdist(f_locs, c_locs)  # [i, j] => distance(facility i -> customer j)

    return f_caps, f_costs, c_dems, f_locs, c_locs, dists


def plot_sol(sol, f_locs, c_locs, title=None):
    plt.figure()

    used_f_locs = f_locs[np.unique(sol)]
    plt.scatter(used_f_locs[:, 0], used_f_locs[:, 1], color="red", s=10)
    plt.scatter(c_locs[:, 0], c_locs[:, 1], color="black", s=1, alpha=0.5)
    for cus, fac in enumerate(sol):
        plt.plot(
            [f_locs[fac][0], c_locs[cus][0]],
            [f_locs[fac][1], c_locs[cus][1]],
            lw=0.1,
            c="black",
            alpha=0.25,
        )
    if title is not None:
        plt.title(title)


class MyIncumbentUpdater(mip.IncumbentUpdater):
    def __init__(self, m: mip.Model, namespace: Dict[str, restore_vars.RestoreMip]):
        super().__init__(m)
        self.namespace = namespace

    def update_incumbent(self, objective_value, solution):
        print(f"incumbent callback")
        # print(objective_value)
        # print(solution)
        ns = self.namespace
        for k in ns.keys():
            print(f"k:")
            print(f"\t{ns[k].restore(solution)}")


def solve_mip(f_caps, c_dems, f_costs, dists) -> np.ndarray:

    nfac, ncus = dists.shape

    m = mip.Model(solver_name=mip.CBC)

    # Bool whether facility i supplies customer j
    ass = [[m.add_var(var_type=mip.BINARY) for _ in range(ncus)] for _ in range(nfac)]
    # Whether each facility is enabled or closed.
    enabled = [m.add_var(var_type=mip.BINARY) for _ in range(nfac)]

    # Each facility must not supply more demand than it has capacity.
    for fac in range(nfac):
        m += mip.xsum([x * c_dems[i] for i, x in enumerate(ass[fac])]) <= f_caps[fac]

    # Each customer must be served by exactly one facility.
    for cus in range(ncus):
        m += mip.xsum([ass[fac][cus] for fac in range(nfac)]) == 1

    # Link whether a facility is open to the assignment array.
    for fac, cus in product(range(nfac), range(ncus)):
        m += enabled[fac] >= ass[fac][cus]

    total_dist = mip.xsum(
        ass[fac][cus] * dists[fac, cus]
        for fac, cus in product(range(nfac), range(ncus))
    )
    total_fac_cost = mip.xsum(enabled[fac] * f_costs[fac] for fac in range(nfac))

    m.objective = mip.minimize(total_dist + total_fac_cost)

    namespace = {
        "ass": restore_vars.Restore2DList(ass),
        "enabled": restore_vars.Restore1DList(enabled),
    }
    m.incumbent_updater = MyIncumbentUpdater(m, namespace)

    status = m.optimize(max_seconds=600)
    if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        raise Exception(f"status: {status}")
    else:
        print(f"status: {status}")

    sol_ass = np.array(
        [[ass[fac][cus].x for cus in range(ncus)] for fac in range(nfac)]
    )

    sol = np.ones(ncus, np.int32) * np.nan
    for cus in range(ncus):
        sol[cus] = np.argmax(sol_ass[:, cus])

    sol = sol.astype(np.int32)
    return sol


def main():
    f = "./data/problems/fl_25_2"
    f_caps, f_costs, c_dems, f_locs, c_locs, dists = load_problem(f)
    sol = solve_mip(f_caps, c_dems, f_costs, dists)

    plot_sol(sol, f_locs, c_locs)
    plt.show(block=True)


if __name__ == "__main__":
    main()
