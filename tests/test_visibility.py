from __future__ import annotations


def test_siblings_cannot_see_each_other(kernel, memory_factory, scope) -> None:
    root = kernel.begin_transaction(scope)
    child_one = kernel.begin_transaction(scope, parent_tx_id=root.tx_id)
    child_two = kernel.begin_transaction(scope, parent_tx_id=root.tx_id)

    parent_item = kernel.stage_memory_item(root.tx_id, memory_factory(title="Parent memory"))
    child_one_item = kernel.stage_memory_item(child_one.tx_id, memory_factory(title="Child one"))
    child_two_item = kernel.stage_memory_item(child_two.tx_id, memory_factory(title="Child two"))

    visible_one = {item.memory_id for item in kernel.list_visible_memory(child_one.tx_id)}
    visible_two = {item.memory_id for item in kernel.list_visible_memory(child_two.tx_id)}
    visible_root = {item.memory_id for item in kernel.list_visible_memory(root.tx_id)}

    assert visible_one == {parent_item.memory_id, child_one_item.memory_id}
    assert visible_two == {parent_item.memory_id, child_two_item.memory_id}
    assert visible_root == {parent_item.memory_id}
