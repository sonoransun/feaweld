#!/usr/bin/env python3
"""Generate Mermaid diagrams for the feaweld advanced-concepts documentation.

Produces 8 `.mmd` files under `docs/diagrams/`. Each is a text mermaid
source that GitHub Markdown renders natively inside triple-backtick
`mermaid` code fences. Where possible the diagram body is generated
from live introspection of the current feaweld code so the diagrams
stay in sync after refactors.

Usage:
    python scripts/generate_docs_mermaid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

DIAGRAMS_DIR = _ROOT / "docs" / "diagrams"


def _write(name: str, body: str) -> Path:
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGRAMS_DIR / f"{name}.mmd"
    path.write_text(body.strip() + "\n")
    return path


def d01_architecture_pipeline() -> str:
    return """
flowchart TD
    YAML[AnalysisCase YAML] --> LOAD[load_case]
    LOAD --> MAT[core.materials.Material]
    MAT --> GEO[geometry.joints / joints3d / spline_joints / fastener_welds]
    GEO --> DEFECT{DefectConfig.enabled?}
    DEFECT -- yes --> POP[defects.population.sample_iso5817_population]
    POP --> INSERT[defects.insertion.insert_all]
    DEFECT -- no --> MESH
    INSERT --> MESH[mesh.generator.generate_mesh]
    MESH --> BACKEND{SolverConfig.backend}
    BACKEND -->|fenics| FE[FEniCSBackend]
    BACKEND -->|calculix| CC[CalculiXBackend]
    BACKEND -->|jax| JB[JAXBackend]
    BACKEND -->|neural| NB[NeuralBackend]
    FE --> FEA[FEAResults]
    CC --> FEA
    JB --> FEA
    NB --> FEA
    FEA --> POST[postprocess.hotspot / dong / linearization / ...]
    POST --> MA{multiaxial_criterion != none?}
    MA -- yes --> MAX[postprocess.multiaxial.findley / dang_van / ...]
    MAX --> FATIGUE
    MA -- no --> FATIGUE[fatigue.rainflow + miner + sn_curves]
    FATIGUE --> UQ[_apply_fatigue_uq: lognormal bands]
    UQ --> K{compute_k_factors?}
    K -- yes --> JI[fracture.j_integral_2d]
    K -- no --> REPORT
    JI --> REPORT[pipeline.report.generate_report]
    REPORT --> OUT[WorkflowResult + HTML report]
"""


def d02_solver_backend_hierarchy() -> str:
    try:
        from feaweld.solver.backend import SolverBackend  # noqa: F401
        from feaweld.solver.jax_backend import JAXBackend  # noqa: F401
        from feaweld.solver.neural_backend import NeuralBackend  # noqa: F401
        # Touch the fenics and calculix backends so __subclasses__ includes them.
        try:
            from feaweld.solver.fenics_backend import FEniCSBackend  # noqa: F401
        except Exception:
            pass
        try:
            from feaweld.solver.calculix_backend import CalculiXBackend  # noqa: F401
        except Exception:
            pass
        subs = sorted({c.__name__ for c in SolverBackend.__subclasses__()})
    except Exception:
        subs = ["FEniCSBackend", "CalculiXBackend", "JAXBackend", "NeuralBackend"]

    lines = [
        "classDiagram",
        "    class SolverBackend {",
        "        <<abstract>>",
        "        +solve_static(mesh, material, load_case, temperature)",
        "        +solve_thermal_steady(mesh, material, load_case)",
        "        +solve_thermal_transient(mesh, material, load_case, time_steps)",
        "        +solve_coupled(mesh, material, mech_lc, thermal_lc, time_steps)",
        "    }",
    ]
    for name in subs:
        lines.append(f"    class {name}")
        lines.append(f"    SolverBackend <|-- {name}")
    lines += [
        "    class JAXConstitutiveModel {",
        "        <<protocol>>",
        "        +stress(strain)",
        "        +tangent(strain)",
        "    }",
        "    JAXBackend o-- JAXConstitutiveModel : uses",
    ]
    return "\n".join(lines)


def d03_constitutive_hierarchy() -> str:
    return """
classDiagram
    class ConstitutiveModel {
        <<abstract>>
        +stress(strain, state)
        +tangent(strain, state)
    }
    class LinearElastic
    class J2Plastic
    class TemperatureDependent

    ConstitutiveModel <|-- LinearElastic
    ConstitutiveModel <|-- J2Plastic
    ConstitutiveModel <|-- TemperatureDependent

    class JAXConstitutiveModel {
        <<protocol>>
        +stress(strain)
        +tangent(strain)
    }
    class JAXLinearElastic
    class JAXJ2Plasticity
    class JAXCrystalPlasticity

    JAXConstitutiveModel <|.. JAXLinearElastic
    JAXConstitutiveModel <|.. JAXJ2Plasticity
    JAXConstitutiveModel <|.. JAXCrystalPlasticity

    LinearElastic ..> JAXLinearElastic : mirror
    J2Plastic ..> JAXJ2Plasticity : mirror
"""


def d04_neural_operator_pipeline() -> str:
    return """
flowchart LR
    A[AnalysisCase template] --> S[pipeline.study.Study<br/>random parameter sweep]
    S --> GT{ground truth backend}
    GT --> JAX[JAXBackend]
    GT --> FE[FEniCSBackend]
    JAX --> DATA[(X, y) training set<br/>load params to displacement field]
    FE --> DATA
    DATA --> TR[Flax DeepONet<br/>branch + trunk networks]
    TR --> LOSS[MSE training<br/>optax.adam]
    LOSS --> SAVED[params.msgpack + meta.json<br/>mesh_hash stamped]
    SAVED --> NB[NeuralBackend.load_model]
    NB --> INFER[solve_static in ~10 ms<br/>mesh hash guard]
"""


def d05_active_learning_loop() -> str:
    return """
stateDiagram-v2
    [*] --> InitialLHS : n_initial random samples
    InitialLHS --> Evaluate
    Evaluate --> FitSurrogate : BayesianFatigueSurrogate.fit
    FitSurrogate --> Acquire : max_variance / EI / random
    Acquire --> PickNext : argmax acquisition
    PickNext --> Evaluate : runner(case)
    Evaluate --> Check : enough iterations?
    Check --> Acquire : no
    Check --> [*] : yes, return ActiveLearningResults
"""


def d06_digital_twin_bayesian_loop() -> str:
    return """
stateDiagram-v2
    [*] --> Ingest : MQTT / OPC-UA stream
    Ingest --> Observe : strain_gauge reading
    Observe --> Assimilate : CrackEnKF.predict
    Assimilate --> Update : CrackEnKF.update
    Update --> Posterior : mean crack length and std
    Posterior --> Check : std exceeds alarm threshold?
    Check --> Alert : yes, dashboard.trigger
    Check --> Ingest : no, continue streaming
    Alert --> [*]
"""


def d07_defect_insertion_workflow() -> str:
    return """
flowchart TD
    C[DefectConfig.enabled] --> S{explicit_defects empty?}
    S -- yes --> DRAW[sample_iso5817_population<br/>level B / C / D]
    S -- no --> LIST[Use explicit defect list]
    DRAW --> VAL[defects.population.validate_population]
    VAL --> BAD{violations?}
    BAD -- yes --> REDRAW[Redraw up to 10 attempts]
    REDRAW --> VAL
    BAD -- no --> LIST
    LIST --> JOINT[Base joint build in Gmsh OCC]
    JOINT --> INSERT[defects.insertion.insert_all]
    INSERT --> PORE[insert_pore: boolean cut sphere]
    INSERT --> SLAG[insert_slag_inclusion: dilate + fragment]
    INSERT --> UND[insert_undercut: sweep + cut]
    INSERT --> LOF[insert_lack_of_fusion: planar fragment]
    INSERT --> CRACK[insert_surface_crack: semi-ellipsoid cut]
    PORE --> TAG[Physical groups tagged]
    SLAG --> TAG
    UND --> TAG
    LOF --> TAG
    CRACK --> TAG
    TAG --> CLEAN[removeAllDuplicates + synchronize]
    CLEAN --> MESH[generate_mesh]
    MESH --> KD[fatigue.knockdown.defect_knockdown]
    KD --> OUT[extensions defects population + worst_fat]
"""


def d08_multipass_welding_sequence() -> str:
    return """
sequenceDiagram
    participant G as geometry.multipass
    participant T as solver.thermal.MultiPassHeatSource
    participant M as solver.mechanical
    participant R as residual stress

    G->>T: build_multipass_joint(base, sequence, path, profiles)
    Note over G,T: root + fill + cap beads as physical groups
    T->>T: Root pass t in [0, d1)<br/>Goldak source 1
    T->>M: temperature field
    M->>R: thermal stress + plasticity
    T->>T: Interpass cooldown
    T->>T: Fill pass t in [d1+gap, ...)<br/>Goldak source 2
    T->>M: temperature field
    M->>R: accumulate residual stress
    T->>T: Cap pass<br/>Goldak source N
    T->>M: temperature field
    M->>R: final residual stress
    R->>G: activate next bead (birth / death)
    Note over G,R: Post-weld heat treatment simulate_pwht
"""


_GENERATORS = {
    "architecture_pipeline": d01_architecture_pipeline,
    "solver_backend_hierarchy": d02_solver_backend_hierarchy,
    "constitutive_hierarchy": d03_constitutive_hierarchy,
    "neural_operator_pipeline": d04_neural_operator_pipeline,
    "active_learning_loop": d05_active_learning_loop,
    "digital_twin_bayesian_loop": d06_digital_twin_bayesian_loop,
    "defect_insertion_workflow": d07_defect_insertion_workflow,
    "multipass_welding_sequence": d08_multipass_welding_sequence,
}


def main() -> list[Path]:
    paths = []
    for name, gen in _GENERATORS.items():
        path = _write(name, gen())
        print(f"wrote {path.relative_to(_ROOT)} ({path.stat().st_size} bytes)")
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
