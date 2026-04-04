"""End-to-end fillet T-joint analysis example.

Demonstrates the full feaweld pipeline: geometry → mesh → solve →
post-process → fatigue assessment → visualization.
"""

from feaweld.pipeline.workflow import (
    AnalysisCase, MaterialConfig, GeometryConfig, MeshConfig,
    SolverConfig, LoadConfig, PostProcessConfig, run_analysis,
)
from feaweld.pipeline.report import generate_report
from feaweld.core.types import JointType, SolverType, StressMethod


def main():
    case = AnalysisCase(
        name="fillet_t_joint_example",
        description="Fillet welded T-joint under axial tension",
        material=MaterialConfig(
            base_metal="A36",
            weld_metal="E70XX",
            haz="A36",
            temperature=20.0,
        ),
        geometry=GeometryConfig(
            joint_type=JointType.FILLET_T,
            base_width=200.0,       # mm
            base_thickness=20.0,    # mm
            web_height=100.0,       # mm
            web_thickness=10.0,     # mm
            weld_leg_size=8.0,      # mm
            length=1.0,             # quasi-2D
        ),
        mesh=MeshConfig(
            global_size=2.0,
            weld_toe_size=0.2,
            element_order=2,
        ),
        solver=SolverConfig(
            solver_type=SolverType.LINEAR_ELASTIC,
            backend="auto",
        ),
        load=LoadConfig(
            axial_force=50000.0,  # 50 kN
        ),
        postprocess=PostProcessConfig(
            stress_methods=[
                StressMethod.HOTSPOT_LINEAR,
                StressMethod.STRUCTURAL_DONG,
                StressMethod.BLODGETT,
            ],
            sn_curve="IIW_FAT90",
            fatigue_assessment=True,
        ),
        output_dir="results/fillet_t_joint",
    )

    print(f"Running analysis: {case.name}")
    result = run_analysis(case)

    if result.success:
        print("Analysis completed successfully!")
        report_path = generate_report(result)
        print(f"Report: {report_path}")
    else:
        print("Analysis completed with errors:")
        for err in result.errors:
            print(f"  - {err}")

    return result


if __name__ == "__main__":
    main()
