"""Pure GPU allocation policy shared by the fleet and its tests."""
import math

GIB = 1024 ** 3


def memory_budget(free_bytes, total_bytes, cap_gib=80):
    return min(int(free_bytes * .9), int(cap_gib * GIB))


def tensor_parallel_size(weight_bytes, budgets, max_tp=None):
    """Reserve runtime space as well as weights; prefer independent replicas."""
    required = int(weight_bytes * 1.2) + 2 * GIB
    available = len(budgets) if max_tp is None else min(len(budgets), max_tp)
    for size in (1, 2, 4, 8):
        if size <= available and required <= min(budgets[:size]) * size:
            return size
    return None


def next_batch_size(current, peak_bytes, baseline_bytes, budget_bytes, ceiling=1024):
    per_example = max((peak_bytes - baseline_bytes) / max(current, 1), 1)
    capacity = max(1, math.floor(max(0, budget_bytes - baseline_bytes) / per_example))
    return max(1, min(ceiling, current * 2, capacity))
