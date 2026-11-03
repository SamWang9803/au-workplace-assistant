"""Tests for reciprocal rank fusion."""
from rag.retrieve import rrf_merge


def test_item_in_both_lists_outranks_items_in_one():
    """A chunk found by both searches beats chunks that topped only one search."""
    assert rrf_merge([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_ties_keep_first_seen_order():
    """Chunks with equal scores stay in the order they were first seen."""
    assert rrf_merge([[7], [9]]) == [7, 9]
    assert rrf_merge([[9], [7]]) == [9, 7]


def test_single_list_order_is_preserved_and_empty_input_is_empty():
    """One list comes back in its own order, and no candidates give no result."""
    assert rrf_merge([[3, 1, 2]]) == [3, 1, 2]
    assert rrf_merge([[], []]) == []
