"""Lazy bridge to packaged HSSD RIR scheduling; native rendering is optional."""
import importlib


def _tools(name):
    return importlib.import_module("avfploc.preprocessing.hssd." + name)


def positive_tools():
    return _tools("hssd_positive_pairs")


def supervisor_tools():
    return _tools("hssd_rir_supervisor")
