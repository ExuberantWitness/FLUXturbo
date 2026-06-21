"""delta/probe.py — thin reuse of cc_wm_demo/extract.py probing primitives."""
import paths  # noqa: F401  (sys.path setup)
import extract as demo_extract  # cc_wm_demo/extract.py

probe_quantity = demo_extract.probe_quantity
probe_all = demo_extract.probe_all
probe_temporal = demo_extract.probe_temporal
pca_components = demo_extract.pca_components
