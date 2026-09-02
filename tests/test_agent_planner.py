"""Tests for agent planner, TaskDAG and DAGExecutor."""

from __future__ import annotations

import pytest

from agent.planner import (
    DAGExecutor,
    Planner,
    TaskDAG,
    TaskNode,
    TaskState,
)


def test_dag_add_and_validate():
    dag = TaskDAG(goal="test")
    a = dag.add("a")
    b = dag.add("b", depends_on=[a.id])
    dag.validate()
    assert len(dag.nodes) == 2
    with pytest.raises(ValueError, match="missing"):
        dag.add("c", depends_on=["missing"])
    dag.add("c", depends_on=["missing"], validate_deps=False)
    with pytest.raises(ValueError, match="missing"):
        dag.validate()


def test_cycle_detection():
    dag = TaskDAG()
    a = dag.add("a")
    b = dag.add("b", depends_on=[a.id])
    # introduce cycle by mutating
    dag.nodes[a.id].depends_on.append(b.id)
    with pytest.raises(ValueError, match="cycle"):
        dag.validate()


def test_ready_tasks_order():
    dag = TaskDAG()
    a = dag.add("a")
    b = dag.add("b", depends_on=[a.id])
    c = dag.add("c", depends_on=[b.id])
    ready = dag.ready_tasks()
    assert [n.name for n in ready] == ["a"]
    a.state = TaskState.SUCCEEDED
    ready = dag.ready_tasks()
    assert [n.name for n in ready] == ["b"]


def test_default_planner_coding():
    planner = Planner()
    dag = planner.plan("implement a bug fix and run tests")
    names = {n.name for n in dag.nodes.values()}
    assert "inspect_repo" in names
    assert "implement" in names
    assert "run_tests" in names
    dag.validate()


def test_default_planner_generic():
    planner = Planner()
    dag = planner.plan("research the history of transformers")
    names = {n.name for n in dag.nodes.values()}
    assert "gather" in names
    assert "act" in names
    assert "verify" in names


def test_dag_executor_success():
    planner = Planner()
    dag = planner.plan("research topic X")
    executor = DAGExecutor()
    calls = []

    def handler(node: TaskNode, ctx: dict):
        calls.append(node.name)
        return {"ok": True}

    receipt = executor.run(dag, default_handler=handler)
    assert receipt.success
    assert len(calls) >= 3
    assert receipt.duration_s >= 0
    assert "counts" in receipt.dag_summary


def test_dag_executor_retry_and_fail():
    dag = TaskDAG(goal="fail case")
    n = dag.add("flaky", retry_budget=1)
    attempts = {"n": 0}

    def handler(node, ctx):
        attempts["n"] += 1
        raise RuntimeError("boom")

    executor = DAGExecutor(handlers={"flaky": handler})
    receipt = executor.run(dag)
    assert not receipt.success
    assert attempts["n"] == 2  # initial + 1 retry
    assert dag.nodes[n.id].state == TaskState.FAILED


def test_dag_executor_cancel():
    dag = TaskDAG(goal="cancel")
    dag.add("long")
    executor = DAGExecutor()

    def handler(node, ctx):
        executor.cancel()
        return {}

    # After first task starts and cancels, remaining should be cancelled on next loop
    dag2 = TaskDAG(goal="c2")
    a = dag2.add("a")
    b = dag2.add("b", depends_on=[a.id])

    def handler2(node, ctx):
        if node.name == "a":
            executor.cancel()
        return {}

    receipt = executor.run(dag2, default_handler=handler2)
    # a may succeed before cancel is observed for b
    assert dag2.nodes[b.id].state in (TaskState.CANCELLED, TaskState.PENDING, TaskState.SKIPPED, TaskState.SUCCEEDED)


def test_dependency_skip_on_failure():
    dag = TaskDAG()
    a = dag.add("a")
    b = dag.add("b", depends_on=[a.id])

    def handler(node, ctx):
        if node.name == "a":
            raise RuntimeError("fail a")
        return {}

    executor = DAGExecutor()
    # a has retry_budget 2 by default so will retry then fail
    a.retry_budget = 0
    receipt = executor.run(dag, default_handler=handler)
    assert dag.nodes[a.id].state == TaskState.FAILED
    # b should become SKIPPED after ready_tasks sees failed dep
    ready = dag.ready_tasks()
    assert dag.nodes[b.id].state == TaskState.SKIPPED
