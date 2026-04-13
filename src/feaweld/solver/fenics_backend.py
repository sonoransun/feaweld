"""FEniCSx (DOLFINx) solver backend.

All DOLFINx / PETSc imports are deferred to function bodies so that the
module can be imported even when FEniCSx is not installed.  An
``ImportError`` is raised only when a solve method is actually called.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend


def _require_dolfinx() -> None:
    """Raise a clear error if DOLFINx is not available."""
    try:
        import dolfinx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FEniCSx (dolfinx) is required for the FEniCS backend. "
            "Install with: pip install feaweld[fenics]"
        ) from exc


def _femesh_to_dolfinx(mesh: FEMesh) -> Any:
    """Convert an :class:`FEMesh` to a DOLFINx mesh object.

    Parameters
    ----------
    mesh : FEMesh
        Solver-agnostic mesh.

    Returns
    -------
    dolfinx.mesh.Mesh
        DOLFINx mesh object.
    """
    import dolfinx.mesh
    from mpi4py import MPI
    import basix

    ndim = mesh.ndim
    n_nodes_per_elem = mesh.elements.shape[1]

    # Map element type to basix cell type
    cell_map = {
        ElementType.TRI3: (basix.CellType.triangle, 1),
        ElementType.TRI6: (basix.CellType.triangle, 2),
        ElementType.QUAD4: (basix.CellType.quadrilateral, 1),
        ElementType.QUAD8: (basix.CellType.quadrilateral, 2),
        ElementType.TET4: (basix.CellType.tetrahedron, 1),
        ElementType.TET10: (basix.CellType.tetrahedron, 2),
        ElementType.HEX8: (basix.CellType.hexahedron, 1),
        ElementType.HEX20: (basix.CellType.hexahedron, 2),
    }

    if mesh.element_type not in cell_map:
        raise ValueError(f"Unsupported element type for FEniCS: {mesh.element_type}")

    cell_type, degree = cell_map[mesh.element_type]

    # Ensure 3D coordinates
    coords = mesh.nodes
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    # Create basix element for coordinate mapping
    coord_element = basix.ufl.element(
        basix.ElementFamily.P, cell_type, degree,
        shape=(coords.shape[1],),
    )

    ufl_domain = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        mesh.elements.astype(np.int64),
        coords,
        dolfinx.mesh.to_type(str(cell_type)),
    )
    return ufl_domain


class FEniCSBackend(SolverBackend):
    """FEA solver backend using FEniCSx / DOLFINx.

    All FEniCSx imports happen inside method bodies to allow graceful
    degradation when the library is not installed.
    """

    def solve_static(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> FEAResults:
        """Linear elastic static solve using FEniCSx."""
        _require_dolfinx()

        import dolfinx
        import dolfinx.fem
        import dolfinx.fem.petsc
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc

        # Material properties at given temperature
        lam = material.lame_lambda(temperature)
        mu = material.lame_mu(temperature)

        # Convert mesh
        domain = _femesh_to_dolfinx(mesh)
        gdim = domain.geometry.dim

        # Function space for displacement
        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Strain and stress
        def epsilon(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w):
            eps = epsilon(w)
            return lam * ufl.tr(eps) * ufl.Identity(gdim) + 2.0 * mu * eps

        # Bilinear form
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

        # Linear form: body forces and surface tractions
        f_body = dolfinx.fem.Constant(domain, np.zeros(gdim, dtype=PETSc.ScalarType))
        L = ufl.inner(f_body, v) * ufl.dx

        # Apply loads from load_case
        for load_bc in load_case.loads:
            if load_bc.bc_type == LoadType.FORCE and load_bc.direction is not None:
                traction = load_bc.values[0] * load_bc.direction[:gdim]
                t_const = dolfinx.fem.Constant(
                    domain, traction.astype(PETSc.ScalarType)
                )
                L += ufl.inner(t_const, v) * ufl.ds

        # Boundary conditions (Dirichlet)
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.DISPLACEMENT:
                u_bc = dolfinx.fem.Constant(
                    domain,
                    constraint.values[:gdim].astype(PETSc.ScalarType),
                )
                # Find boundary DOFs from node set
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, 0, node_ids.astype(np.int32)
                    )
                else:
                    # Fall back: fix all boundary facets
                    boundary_facets = dolfinx.mesh.exterior_facet_indices(
                        domain.topology
                    )
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, domain.topology.dim - 1, boundary_facets
                    )
                bc = dolfinx.fem.dirichletbc(u_bc, dofs, V)
                bcs.append(bc)

        # Assemble and solve
        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()

        # Extract displacement at mesh nodes
        n_nodes = mesh.n_nodes
        disp = np.zeros((n_nodes, 3))
        coords = uh.x.array.reshape(-1, gdim)
        disp[:coords.shape[0], :gdim] = coords

        # Compute stress field via projection to DG space
        W = dolfinx.fem.functionspace(domain, ("DG", 0, (gdim, gdim)))
        sigma_expr = sigma(uh)
        stress_func = dolfinx.fem.Function(W)
        stress_expr = dolfinx.fem.Expression(
            sigma_expr, W.element.interpolation_points()
        )
        stress_func.interpolate(stress_expr)

        # Extract stress in Voigt notation
        stress_vals = stress_func.x.array.reshape(-1, gdim, gdim)
        n_cells = stress_vals.shape[0]
        voigt_stress = np.zeros((n_cells, 6))
        voigt_stress[:, 0] = stress_vals[:, 0, 0]  # sigma_xx
        voigt_stress[:, 1] = stress_vals[:, 1, 1]  # sigma_yy
        if gdim == 3:
            voigt_stress[:, 2] = stress_vals[:, 2, 2]  # sigma_zz
            voigt_stress[:, 3] = stress_vals[:, 0, 1]  # tau_xy
            voigt_stress[:, 4] = stress_vals[:, 1, 2]  # tau_yz
            voigt_stress[:, 5] = stress_vals[:, 0, 2]  # tau_xz
        else:
            voigt_stress[:, 3] = stress_vals[:, 0, 1]  # tau_xy

        stress_field = StressField(values=voigt_stress, location="gauss_points")

        return FEAResults(
            mesh=mesh,
            displacement=disp,
            stress=stress_field,
            metadata={"solver": "fenics", "temperature": temperature},
        )

    def solve_thermal_steady(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
    ) -> FEAResults:
        """Steady-state thermal solve using FEniCSx."""
        _require_dolfinx()

        import dolfinx
        import dolfinx.fem
        import dolfinx.fem.petsc
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc

        domain = _femesh_to_dolfinx(mesh)

        # Scalar function space for temperature
        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

        T = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Use conductivity at room temperature as a starting point
        k_val = material.k(20.0)
        k = dolfinx.fem.Constant(domain, PETSc.ScalarType(k_val))

        # Weak form: integral k * grad(T) . grad(v) dx = integral Q * v dx
        a = k * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

        # Source and boundary terms
        Q = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
        L = Q * v * ufl.dx

        # Process loads: heat flux, convection
        for load_bc in load_case.loads:
            if load_bc.bc_type == LoadType.HEAT_FLUX:
                q_val = load_bc.values[0]
                q = dolfinx.fem.Constant(domain, PETSc.ScalarType(q_val))
                L += q * v * ufl.ds
            elif load_bc.bc_type == LoadType.CONVECTION:
                h_conv = load_bc.values[0]
                T_amb = load_bc.values[1] if len(load_bc.values) > 1 else 20.0
                h = dolfinx.fem.Constant(domain, PETSc.ScalarType(h_conv))
                T_a = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_amb))
                a += h * T * v * ufl.ds
                L += h * T_a * v * ufl.ds

        # Dirichlet BCs (fixed temperature)
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.TEMPERATURE:
                T_bc_val = constraint.values[0]
                T_bc = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_bc_val))
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, 0, node_ids.astype(np.int32)
                    )
                else:
                    boundary_facets = dolfinx.mesh.exterior_facet_indices(
                        domain.topology
                    )
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, domain.topology.dim - 1, boundary_facets
                    )
                bc = dolfinx.fem.dirichletbc(T_bc, dofs, V)
                bcs.append(bc)

        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        Th = problem.solve()

        temp_array = Th.x.array.copy()
        # Pad to n_nodes if needed
        result_temp = np.full(mesh.n_nodes, 20.0)
        result_temp[: len(temp_array)] = temp_array

        return FEAResults(
            mesh=mesh,
            temperature=result_temp,
            metadata={"solver": "fenics", "analysis": "thermal_steady"},
        )

    def solve_thermal_transient(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        time_steps: NDArray,
        heat_source: object | None = None,
        initial_temperature: NDArray | None = None,
    ) -> FEAResults:
        """Transient thermal solve using backward Euler time stepping."""
        _require_dolfinx()

        import dolfinx
        import dolfinx.fem
        import dolfinx.fem.petsc
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc

        time_steps = np.asarray(time_steps, dtype=np.float64)
        domain = _femesh_to_dolfinx(mesh)

        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

        T = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Previous temperature
        T_n = dolfinx.fem.Function(V)
        if initial_temperature is not None:
            init_t = np.asarray(initial_temperature, dtype=np.float64)
            T_n.x.array[:len(init_t)] = init_t[:len(T_n.x.array)]
        else:
            T_n.x.array[:] = 20.0  # initial condition

        # Material properties (use room-temperature values for simplicity;
        # for full nonlinearity, one would iterate within each time step)
        rho = material.density
        cp = material.cp(20.0)
        k_val = material.k(20.0)

        rho_cp = dolfinx.fem.Constant(domain, PETSc.ScalarType(rho * cp))
        k = dolfinx.fem.Constant(domain, PETSc.ScalarType(k_val))

        # Time step size (will be updated each step)
        dt_const = dolfinx.fem.Constant(domain, PETSc.ScalarType(1.0))

        # Heat source as a Function (updated at each time step if moving source)
        Q = dolfinx.fem.Function(V)
        Q.x.array[:] = 0.0

        # Backward Euler weak form:
        # rho*cp * (T - T_n)/dt * v dx + k * grad(T) . grad(v) dx = Q * v dx
        a = (rho_cp / dt_const) * T * v * ufl.dx + k * ufl.inner(
            ufl.grad(T), ufl.grad(v)
        ) * ufl.dx
        L = (rho_cp / dt_const) * T_n * v * ufl.dx + Q * v * ufl.dx

        # Convection / flux from load case
        for load_bc in load_case.loads:
            if load_bc.bc_type == LoadType.CONVECTION:
                h_conv = load_bc.values[0]
                T_amb = load_bc.values[1] if len(load_bc.values) > 1 else 20.0
                h = dolfinx.fem.Constant(domain, PETSc.ScalarType(h_conv))
                T_a = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_amb))
                a += h * T * v * ufl.ds
                L += h * T_a * v * ufl.ds
            elif load_bc.bc_type == LoadType.HEAT_FLUX:
                q_val = load_bc.values[0]
                q = dolfinx.fem.Constant(domain, PETSc.ScalarType(q_val))
                L += q * v * ufl.ds

        # Dirichlet BCs
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.TEMPERATURE:
                T_bc_val = constraint.values[0]
                T_bc = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_bc_val))
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, 0, node_ids.astype(np.int32)
                    )
                else:
                    boundary_facets = dolfinx.mesh.exterior_facet_indices(
                        domain.topology
                    )
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, domain.topology.dim - 1, boundary_facets
                    )
                bc = dolfinx.fem.dirichletbc(T_bc, dofs, V)
                bcs.append(bc)

        # Compile forms
        bilinear_form = dolfinx.fem.form(a)
        linear_form = dolfinx.fem.form(L)

        # Storage for temperature history
        n_steps = len(time_steps)
        n_dofs = len(T_n.x.array)
        temp_history = np.zeros((n_steps, mesh.n_nodes))
        temp_history[0, :n_dofs] = T_n.x.array

        # PETSc solver
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        T_sol = dolfinx.fem.Function(V)

        for step_idx in range(1, n_steps):
            dt_val = time_steps[step_idx] - time_steps[step_idx - 1]
            dt_const.value = dt_val
            t_current = time_steps[step_idx]

            # Update heat source if provided
            if heat_source is not None and hasattr(heat_source, "evaluate"):
                x_coords = domain.geometry.x
                q_vals = heat_source.evaluate(
                    x_coords[:, 0], x_coords[:, 1], x_coords[:, 2], t_current
                )
                Q.x.array[: len(q_vals)] = q_vals

            # Assemble
            A = dolfinx.fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
            A.assemble()
            b = dolfinx.fem.petsc.assemble_vector(linear_form)
            dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, bcs)

            solver.setOperators(A)
            solver.solve(b, T_sol.x.petsc_vec)

            # Update T_n for next step
            T_n.x.array[:] = T_sol.x.array

            temp_history[step_idx, :n_dofs] = T_sol.x.array

        return FEAResults(
            mesh=mesh,
            temperature=temp_history,
            time_steps=time_steps,
            metadata={"solver": "fenics", "analysis": "thermal_transient"},
        )

    def solve_coupled(
        self,
        mesh: FEMesh,
        material: Material,
        mechanical_lc: LoadCase,
        thermal_lc: LoadCase,
        time_steps: NDArray,
    ) -> FEAResults:
        """Sequential thermomechanical coupling: thermal first, then mechanical."""
        from feaweld.solver.thermomechanical import sequential_coupled_solve

        return sequential_coupled_solve(
            backend=self,
            mesh=mesh,
            material=material,
            thermal_lc=thermal_lc,
            mechanical_lc=mechanical_lc,
            time_steps=time_steps,
        )
