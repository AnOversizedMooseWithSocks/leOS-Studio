"""tests/test_r25.py -- unreachable graph nodes must be named (R26 find).

Grading The Fifth Labyrinth, a client posted its wiring as a separate
'wires' list -- a key /api/graph does not read. Every node sat unwired,
the Output node fell back to the raw composite, and two rounds of
'grades' served untouched pixels behind an ok:true. The R24 fix names
unknown TYPES; this one names unknown SHAPE: any node that cannot reach
the Output node is listed in the POST response's 'unreachable' warning.
"""
import numpy as np


def _graph_doc():
    from lestudio import Document, NodeGraph
    d = Document(64, 48)
    d.layers[0].pixels[..., :3] = 0.4
    d.layers[0].pixels[..., 3] = 1.0
    return d, NodeGraph(d)


def test_r25_orphans_are_named():
    d, g = _graph_doc()
    g.set_graph([
        {"id": "src", "type": "Media in", "params": {}, "inputs": {}},
        {"id": "grd", "type": "Grade", "params": {"gain": 2.0},
         "inputs": {}},
        {"id": "out", "type": "Output", "params": {}, "inputs": {}},
    ])
    assert g.unreachable_nodes() == ["grd", "src"]


def test_r25_wired_graph_has_no_orphans():
    d, g = _graph_doc()
    g.set_graph([
        {"id": "src", "type": "Media in", "params": {}, "inputs": {}},
        {"id": "grd", "type": "Grade", "params": {"gain": 2.0},
         "inputs": {"image": "src"}},
        {"id": "out", "type": "Output", "params": {},
         "inputs": {"image": "grd"}},
    ])
    assert g.unreachable_nodes() == []


def test_r25_socket_suffix_and_list_refs_count_as_wires():
    d, g = _graph_doc()
    g.set_graph([
        {"id": "a", "type": "Grade", "params": {}, "inputs": {}},
        {"id": "b", "type": "Grade", "params": {},
         "inputs": {"image": "a.image"}},
        {"id": "out", "type": "Output", "params": {},
         "inputs": {"image": ["b", "image"]}},
    ])
    assert g.unreachable_nodes() == []


def test_r25_default_graph_stays_quiet():
    """The untouched default (a lone Output) keeps its composite
    convenience -- no orphan warning for the one-node graph."""
    d, g = _graph_doc()
    g.ensure_default()
    assert g.unreachable_nodes() == []
