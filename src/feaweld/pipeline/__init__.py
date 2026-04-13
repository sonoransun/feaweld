"""Pipeline sub-package: workflow orchestration, parametric studies, and comparison.

Modules
-------
workflow
    Main pipeline orchestrator (``run_analysis``, ``AnalysisCase``).
study
    Parametric sweep builder and concurrent execution.
comparison
    Multi-case comparison reporting.
dag
    DAG-based pipeline executor with concurrent batching.
hooks
    Pre/post/error callbacks and built-in observability hooks.
checkpoint
    Checkpoint/restart for long-running pipelines.
distributed
    Dask/Ray backends for cluster-scale study execution.
queue
    SQLite-backed persistent job queue.
ipc
    Shared-memory transport for large arrays between workers.
"""
