# -*- coding: utf-8 -*-
"""Extract cantilever results from Abaqus ODB files.

Runs under Abaqus Python (2.7 in older releases, 3.x in newer ones), not your own Python.
abq_runner.extract() copies this file into the work directory and calls:

    abaqus python abq_extract.py <list.txt> <out.json>

list.txt holds one job per line: "<job name><TAB><odb file>".
"""
from __future__ import print_function

import json
import sys

from odbAccess import openOdb


def find_nset(odb, name):
    asm = odb.rootAssembly
    if name in asm.nodeSets.keys():
        return asm.nodeSets[name]
    for inst in asm.instances.values():
        if name in inst.nodeSets.keys():
            return inst.nodeSets[name]
    raise KeyError("node set %s not found" % name)


def first_label(nset):
    nodes = nset.nodes
    if len(nodes) and not hasattr(nodes[0], "label"):   # assembly set: one list per instance
        nodes = nodes[0]
    return nodes[0].label


def node_vector(frame, var, nset):
    return frame.fieldOutputs[var].getSubset(region=nset).values[0].data


def field_max(frame, var, attr=None):
    if var not in frame.fieldOutputs.keys():
        return None
    best = None
    for v in frame.fieldOutputs[var].values:
        x = getattr(v, attr) if attr else v.data
        if best is None or x > best:
            best = x
    return None if best is None else float(best)


def history_u2(step, label):
    suffix = "." + str(label)
    for name, region in step.historyRegions.items():
        if name.startswith("Node") and name.endswith(suffix):
            if "U2" in region.historyOutputs.keys():
                return [[float(t), float(u)] for t, u in region.historyOutputs["U2"].data]
    return []


def extract(path):
    odb = openOdb(path=path, readOnly=True)
    try:
        tip = find_nset(odb, "TIP_MID")
        root = find_nset(odb, "ROOT_MID")
        label = first_label(tip)
        r = {"ok": True}

        load = odb.steps["LOAD"]
        f = load.frames[-1]
        u = node_vector(f, "U", tip)
        r["load_frac_end"] = float(f.frameValue)
        r["u1_peak"], r["u2_peak"] = float(u[0]), float(u[1])
        r["rf2_root_peak"] = float(node_vector(f, "RF", root)[1])
        r["peeq_max_peak"] = field_max(f, "PEEQ")
        r["mises_max_peak"] = field_max(f, "S", "mises")
        r["curve_load"] = history_u2(load, label)

        if "UNLOAD" in odb.steps.keys():
            un = odb.steps["UNLOAD"]
            g = un.frames[-1]
            u = node_vector(g, "U", tip)
            r["unload_frac_end"] = float(g.frameValue)
            r["u1_res"], r["u2_res"] = float(u[0]), float(u[1])
            r["mises_max_res"] = field_max(g, "S", "mises")
            r["curve_unload"] = history_u2(un, label)
        return r
    finally:
        odb.close()


def main():
    list_file, out_file = sys.argv[1], sys.argv[2]
    results = {}
    with open(list_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            job, odb = line.split("\t")
            try:
                results[job] = extract(odb)
            except Exception as e:  # keep going; the job is reported as failed
                results[job] = {"ok": False, "error": repr(e)}
            print("extracted", job, results[job].get("ok"))
    with open(out_file, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    main()
