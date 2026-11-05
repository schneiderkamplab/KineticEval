from kineticeval.gpu_policy import GIB, memory_budget, tensor_parallel_size, next_batch_size


def test_budget_is_ninety_percent_of_free_not_total():
    assert memory_budget(80*GIB,180*GIB) == 72*GIB
    assert memory_budget(160*GIB,180*GIB) == 80*GIB


def test_tp_uses_smallest_fitting_group_with_runtime_reserve():
    budgets = [80*GIB]*8
    assert tensor_parallel_size(2*GIB,budgets) == 1
    assert tensor_parallel_size(100*GIB,budgets) == 2
    assert tensor_parallel_size(200*GIB,budgets) == 4
    assert tensor_parallel_size(1000*GIB,budgets) is None
    assert tensor_parallel_size(100*GIB,budgets[:1]) is None


def test_batch_adaptation_respects_observed_activation_cost():
    assert next_batch_size(8,10*GIB,2*GIB,20*GIB) == 16
    assert next_batch_size(32,40*GIB,8*GIB,24*GIB) == 16
    assert next_batch_size(1,90*GIB,80*GIB,80*GIB) == 1


def test_headroom_gate_filters_every_gpu_before_placement():
    from kineticeval.gpu_policy import eligible_gpus, GIB
    memory={0:(79*GIB,180*GIB),1:(80*GIB,180*GIB),2:(100*GIB,180*GIB)}
    assert eligible_gpus(memory,{0,1,2},set(),80)==[2,1]
    assert eligible_gpus(memory,{0,1,2},{2},80)==[1]
    assert eligible_gpus(memory,{0},set(),80)==[]
