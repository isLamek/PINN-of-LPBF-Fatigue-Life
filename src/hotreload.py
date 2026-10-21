"""Reload page modules whose source changed since they were imported.

Streamlit reruns app.py from disk on every interaction but keeps imported modules in
memory. After a push, Streamlit Cloud pulls the new files into the running process, and
a page module imported before the pull can keep serving its old text beside new data.
Each module is stamped with the modification time of the file it was loaded from; when
any stamp is missing or out of date, every listed module is reloaded in order, so
pages that import names from src.ui pick up a changed ui as well.
"""
import importlib
import os


def refresh(*modules):
    stale = [m for m in modules
             if getattr(m, "_hot_mtime", None) != os.path.getmtime(m.__file__)]
    if not stale:
        return
    for m in modules:
        importlib.reload(m)
        m._hot_mtime = os.path.getmtime(m.__file__)
