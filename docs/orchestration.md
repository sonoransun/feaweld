# Pipeline Orchestration

feaweld executes analysis cases through a DAG-based pipeline that supports concurrent stage execution, pre/post hooks, checkpoint/restart, distributed scaling, and a persistent job queue.

## DAG Pipeline

The pipeline is modeled as a directed acyclic graph of stages. Independent stages run concurrently in thread-pool batches while dependent stages wait for their inputs.

```mermaid
flowchart TD
    subgraph B0["Batch 0 -- independent"]
        materials[materials]
        defects[defect config]
    end
    subgraph B1["Batch 1"]
        geometry[geometry]
        defect_sample[defect sample]
    end
    subgraph B2["Batch 2"]
        mesh[mesh generation]
    end
    subgraph B3["Batch 3"]
        solve[FEA solve]
    end
    subgraph B4["Batch 4 -- concurrent post-processing"]
        hotspot[hotspot]
        dong[dong]
        nominal[nominal]
        linear[linearization]
        sed[SED]
        notch[notch stress]
    end
    subgraph B5["Batch 5"]
        fatigue[fatigue assessment]
        jint[J-integral]
    end
    subgraph B6["Batch 6"]
        prob[probabilistic]
    end
    report[report generation]

    B0 --> B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> report
```

Each batch boundary is a checkpoint save point. If the process is interrupted (SIGTERM, crash), resuming from a checkpoint skips completed batches.

### Programmatic usage

```python
from feaweld.pipeline.dag import PipelineDAG, PipelineStage, PipelineContext

dag = PipelineDAG()
dag.add(PipelineStage("materials", load_materials))
dag.add(PipelineStage("geometry", build_geometry))
dag.add(PipelineStage("mesh", gen_mesh, depends_on=["geometry"]))
dag.add(PipelineStage("solve", run_solver, depends_on=["mesh", "materials"]))

ctx = PipelineContext(case=my_case)
dag.run(ctx, max_workers=4)
```

## Pipeline Hooks

Hooks let you inject custom logic before and after each stage, or on errors. The built-in `timing_hook()` and `memory_hook()` factories provide observability out of the box.

```mermaid
sequenceDiagram
    participant DAG as PipelineDAG
    participant PRE as pre_stage hooks
    participant STAGE as Stage callable
    participant POST as post_stage hooks
    participant ERR as on_error hooks

    DAG->>PRE: invoke(stage_name, context)
    PRE->>STAGE: stage.callable(context)
    alt success
        STAGE->>POST: invoke(stage_name, context)
    else exception
        STAGE->>ERR: invoke(stage_name, exception)
    end
```

### Example: timing and memory hooks

```python
from feaweld.pipeline.hooks import PipelineHooks, timing_hook, memory_hook

hooks = timing_hook()
hooks.merge(memory_hook())

dag.run(ctx, hooks=hooks)

for stage, dt in hooks._timings.items():
    print(f"{stage}: {dt:.2f}s")
```

## Checkpoint / Restart

Long-running analyses can be checkpointed after each batch. If the process crashes or receives SIGTERM, the analysis resumes from the last completed batch.

```mermaid
stateDiagram-v2
    [*] --> Running : dag.run()
    Running --> Checkpoint : batch completes
    Checkpoint --> Running : next batch
    Running --> Crash : exception / SIGTERM
    Crash --> Resume : run --resume checkpoint/
    Resume --> Running : skip completed stages
    Running --> Done : all stages complete
    Done --> Cleanup : clear_checkpoint()
    Cleanup --> [*]
```

The checkpoint directory contains:

| File | Content |
|------|---------|
| `meta.json` | Completed stage names, config SHA-256 hash, key-type map |
| `*.npz` | Large numpy arrays (mesh nodes, stress fields) |
| `*.pkl` | Non-array Python objects (configs, small results) |

### Usage

```python
from feaweld.pipeline.checkpoint import save_checkpoint, load_checkpoint

# Save after each batch (done automatically by the DAG executor)
save_checkpoint(ctx, Path("checkpoint/"), completed_stages=["materials", "geometry"])

# Resume from checkpoint
ctx, completed = load_checkpoint(Path("checkpoint/"))
dag.run(ctx, skip_stages=completed)
```

## Distributed Execution

Parametric studies can scale beyond a single machine using Dask or Ray clusters.

```mermaid
flowchart LR
    DS[DistributedStudy] --> BE{backend?}
    BE -->|dask| DC[Dask Client]
    BE -->|ray| RC[Ray runtime]
    DC --> S[Scheduler]
    RC --> S
    S --> W1[Worker 1]
    S --> W2[Worker 2]
    S --> W3[Worker N]
    W1 --> RES[results dict]
    W2 --> RES
    W3 --> RES
    RES --> CB[progress callback]
```

### Usage

```python
from feaweld.pipeline.distributed import DistributedStudy

ds = DistributedStudy(backend="dask")
results = ds.run(cases)  # dict[str, WorkflowResult]
```

Or via the CLI with the study backend option:

```bash
feaweld study run study.yaml -j 16 --backend dask
```

Install the distributed extra: `pip install feaweld[distributed]`

## Job Queue

The SQLite-backed job queue provides persistent, priority-based scheduling for analysis jobs. A worker loop claims and executes jobs atomically.

```mermaid
stateDiagram-v2
    [*] --> PENDING : submit(case, priority)
    PENDING --> RUNNING : worker claims
    RUNNING --> COMPLETED : success
    RUNNING --> FAILED : exception
    PENDING --> [*] : cancel()
    COMPLETED --> [*] : purge()
    FAILED --> [*] : purge()
```

### CLI usage

```bash
# Submit jobs with priorities (lower = higher priority)
feaweld queue submit urgent_case.yaml -p 0
feaweld queue submit normal_case.yaml -p 5

# Check queue status
feaweld queue status

# Start a worker (blocks, processes jobs in priority order)
python -c "from feaweld.pipeline.queue import AnalysisJobQueue; AnalysisJobQueue().worker_loop()"
```

## Shared Memory IPC

When running parametric studies with `ProcessPoolExecutor`, large numpy arrays (stress fields, displacement vectors) are transferred between processes. The `SharedResultStore` uses `multiprocessing.shared_memory` for zero-copy transfer instead of pickle serialization.

```mermaid
flowchart LR
    W[Worker Process] -->|"store(array)"| SHM["/dev/shm\nSharedMemory"]
    SHM -->|"SharedArrayMeta\n(name, shape, dtype)"| MAIN[Main Process]
    MAIN -->|"retrieve(meta)"| SHM
    SHM --> ARR[numpy array copy]
```

This is automatic when shared memory is available and falls back to normal pickling on platforms that don't support it.
