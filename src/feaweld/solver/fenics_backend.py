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
    """Convert an [FEMesh][feaweld.core.types.FEMesh] to a DOLFINx mesh object.

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

    if degree == 2 and mesh.element_type in (ElementType.TET10, ElementType.HEX20):
        raise NotImplementedError(
            f"3D second-order cells ({mesh.element_type.value}) are not yet "
            "supported by the FEniCS backend (gmsh/basix mid-node ordering "
            "unverified); use element_order=1 or the CalculiX backend."
        )

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


def _block_dofs_for_nodes(space: Any, mesh: FEMesh, node_ids: NDArray) -> NDArray:
    """Map input-mesh node ids to a DOLFINx space's block-dof indices.

    ``dolfinx.mesh.create_mesh`` reorders vertices relative to the input
    :class:`~feaweld.core.types.FEMesh`, so per-node data cannot be indexed
    by the original node id.  This helper matches the requested nodes'
    coordinates against ``space.tabulate_dof_coordinates()`` with a KD-tree
    and returns, for each requested node, the *block* dof index (one entry
    per node, independent of the number of vector components).  For a scalar
    space the block dof equals the scalar dof; for a vector space of block
    size ``bs`` the scalar dof of component ``c`` is ``block_dof * bs + c``.

    Parameters
    ----------
    space : dolfinx.fem.FunctionSpace
        Target function space (scalar or vector) whose dof coordinates
        supply the match targets.
    mesh : FEMesh
        Input mesh providing the reference node coordinates.
    node_ids : numpy.ndarray
        Input-mesh node indices to locate.

    Returns
    -------
    numpy.ndarray
        Block dof index for each requested node, dtype ``int32``.

    Notes
    -----
    Assumes ``space`` resolves every requested node (true for first-order
    Lagrange spaces on first-order meshes, where the number of block dofs
    equals ``mesh.n_nodes``).  The match is nearest-neighbour, so it is
    robust to the vertex reordering but not to a coarser space than the
    input node set.
    """
    from scipy.spatial import cKDTree

    dof_coords = space.tabulate_dof_coordinates()
    targets = mesh.nodes[node_ids]
    if targets.shape[1] == 2:
        targets = np.column_stack([targets, np.zeros(len(targets))])
    _, block_dofs = cKDTree(dof_coords[:, :3]).query(targets)
    return np.asarray(block_dofs, dtype=np.int32)


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
        """Linear elastic static solve using FEniCSx.

        Assembles a vector-valued elasticity problem with the provided
        material properties and load case, solves the linear system,
        and returns nodal displacements plus element-averaged stress
        and strain.

        Parameters
        ----------
        mesh, material, load_case, temperature
            See [SolverBackend.solve_static][feaweld.solver.backend.SolverBackend.solve_static].

        Raises
        ------
        ImportError
            If ``dolfinx`` is not installed.
        """
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
        tdim = domain.topology.dim

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

        # Apply loads from load_case.  Nodal forces (per-node vectors and
        # scalar force + direction on a known node set) are collected as
        # ``(node_ids, (n, gdim) rows)`` and injected into the RHS after
        # assembly; only tractions and pressures enter the variational form.
        nodal_forces: list[tuple[NDArray, NDArray]] = []
        # Thermoelastic field, retained for stress recovery (thermal eigenstrain).
        temp_field: tuple[Any, float] | None = None
        for load_bc in load_case.loads:
            values = np.asarray(load_bc.values)
            if load_bc.bc_type == LoadType.FORCE and values.ndim == 2:
                # Per-node force vectors (e.g. moment couples): inject to RHS.
                node_ids = mesh.node_sets.get(load_bc.node_set)
                if node_ids is not None:
                    nodal_forces.append((node_ids, values.astype(np.float64)))
            elif load_bc.bc_type == LoadType.FORCE and load_bc.direction is not None:
                # Scalar force magnitude + direction.  CalculiX *CLOAD applies
                # ``F * direction`` to every node in the set (per-node force
                # [N]); mirror that exactly by expanding into per-node force
                # rows and routing through the nodal-force injection path
                # rather than smearing a traction over the whole boundary.
                node_ids = mesh.node_sets.get(load_bc.node_set)
                direction = np.asarray(load_bc.direction, dtype=np.float64)[:gdim]
                mag = float(values.ravel()[0])
                if node_ids is not None:
                    rows = np.tile(mag * direction, (len(node_ids), 1))
                    nodal_forces.append((node_ids, rows))
                else:
                    # Fallback ONLY when the node set is unknown: treat the
                    # value as a uniform surface traction [N/mm^2] over the
                    # entire exterior boundary.  This is dimensionally
                    # different from a nodal force and is a last resort.
                    t_const = dolfinx.fem.Constant(
                        domain, (mag * direction).astype(PETSc.ScalarType)
                    )
                    L += ufl.inner(t_const, v) * ufl.ds
            elif load_bc.bc_type == LoadType.PRESSURE:
                # Pressure acts along the inward surface normal
                n_facet = ufl.FacetNormal(domain)
                p_const = dolfinx.fem.Constant(
                    domain, PETSc.ScalarType(load_bc.values[0])
                )
                L += -p_const * ufl.inner(n_facet, v) * ufl.ds
            elif load_bc.bc_type == LoadType.TEMPERATURE:
                # Thermoelastic load: sigma_th = (3*lam + 2*mu) * alpha * dT
                # Scalar values are uniform absolute temperatures; per-node
                # arrays give the nodal temperature field (20 C reference).
                S = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
                T_fn = dolfinx.fem.Function(S)
                flat = values.ravel().astype(np.float64)
                if flat.shape[0] == mesh.n_nodes:
                    s_dofs = _block_dofs_for_nodes(S, mesh, np.arange(mesh.n_nodes))
                    T_fn.x.array[s_dofs] = flat
                else:
                    T_fn.x.array[:] = flat[0]
                alpha_val = material.alpha(temperature)
                dT = T_fn - 20.0
                L += (
                    (3.0 * lam + 2.0 * mu) * alpha_val * dT
                    * ufl.tr(epsilon(v)) * ufl.dx
                )
                temp_field = (T_fn, alpha_val)

        # Boundary conditions (Dirichlet).  Node-set dofs are matched by
        # coordinate (via ``_block_dofs_for_nodes``) because DOLFINx reorders
        # vertices relative to the input mesh, so the original node ids are
        # not valid DOLFINx vertex indices.
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.DISPLACEMENT:
                values = np.asarray(constraint.values)
                if values.ndim == 2 and constraint.node_set in mesh.node_sets:
                    # Per-node prescribed displacements (submodel cut boundary)
                    node_ids = mesh.node_sets[constraint.node_set]
                    block_dofs = _block_dofs_for_nodes(V, mesh, node_ids)
                    u_fn = dolfinx.fem.Function(V)
                    for bd, row in zip(block_dofs, values):
                        for comp in range(gdim):
                            u_fn.x.array[int(bd) * gdim + comp] = row[comp]
                    bc = dolfinx.fem.dirichletbc(u_fn, block_dofs)
                    bcs.append(bc)
                    continue
                u_bc = dolfinx.fem.Constant(
                    domain,
                    constraint.values[:gdim].astype(PETSc.ScalarType),
                )
                # Find boundary DOFs from node set (coordinate-matched block dofs)
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = _block_dofs_for_nodes(V, mesh, node_ids)
                else:
                    # Fall back: fix all boundary facets
                    domain.topology.create_connectivity(tdim - 1, tdim)
                    boundary_facets = dolfinx.mesh.exterior_facet_indices(
                        domain.topology
                    )
                    dofs = dolfinx.fem.locate_dofs_topological(
                        V, tdim - 1, boundary_facets
                    )
                bc = dolfinx.fem.dirichletbc(u_bc, dofs, V)
                bcs.append(bc)

        # Assemble and solve
        if not nodal_forces:
            problem = dolfinx.fem.petsc.LinearProblem(
                a, L, bcs=bcs,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )
            uh = problem.solve()
        else:
            # Manual assembly so point forces can be added to the RHS vector
            from mpi4py import MPI as _MPI

            a_form = dolfinx.fem.form(a)
            L_form = dolfinx.fem.form(L)
            A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
            A.assemble()
            b = dolfinx.fem.petsc.assemble_vector(L_form)

            for node_ids, rows in nodal_forces:
                block_dofs = _block_dofs_for_nodes(V, mesh, node_ids)
                for bd, row in zip(block_dofs, rows):
                    for comp in range(min(gdim, row.shape[0])):
                        b.setValue(
                            int(bd) * gdim + comp,
                            float(row[comp]),
                            addv=PETSc.InsertMode.ADD_VALUES,
                        )
            b.assemblyBegin()
            b.assemblyEnd()

            dolfinx.fem.petsc.apply_lifting(b, [a_form], [bcs])
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            dolfinx.fem.petsc.set_bc(b, bcs)

            solver = PETSc.KSP().create(_MPI.COMM_WORLD)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            solver.setOperators(A)
            uh = dolfinx.fem.Function(V)
            solver.solve(b, uh.x.petsc_vec)

        # ------------------------------------------------------------------
        # Reorder results into INPUT node / element order.  DOLFINx reorders
        # vertices and cells relative to the FEMesh, so raw dof-order and
        # cell-order arrays would break the solver-agnostic FEAResults
        # contract (CalculiX returns input-node-ordered nodal fields).
        # ------------------------------------------------------------------
        n_nodes = mesh.n_nodes
        block_dofs_all = _block_dofs_for_nodes(V, mesh, np.arange(n_nodes))
        dof_block_values = uh.x.array.reshape(-1, gdim)
        disp = np.zeros((n_nodes, 3))
        disp[:, :gdim] = dof_block_values[block_dofs_all]

        # Stress expression.  When a temperature load was applied, remove the
        # thermal eigenstrain so the reported stress is the true mechanical
        # stress C:(eps(u) - alpha*dT*I), not C:eps(u).
        if temp_field is not None:
            T_fn, alpha_val = temp_field
            eps_th = alpha_val * (T_fn - 20.0)

            def sigma_out(w):
                eps_m = epsilon(w) - eps_th * ufl.Identity(gdim)
                return lam * ufl.tr(eps_m) * ufl.Identity(gdim) + 2.0 * mu * eps_m

            sigma_expr_ufl = sigma_out(uh)
        else:
            sigma_expr_ufl = sigma(uh)

        # Element-constant (DG0) stress projection
        W = dolfinx.fem.functionspace(domain, ("DG", 0, (gdim, gdim)))
        stress_func = dolfinx.fem.Function(W)
        stress_expr = dolfinx.fem.Expression(
            sigma_expr_ufl, W.element.interpolation_points()
        )
        stress_func.interpolate(stress_expr)

        # Extract stress in Voigt notation (one row per DOLFINx cell)
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

        # Map DG0 cell stress back to INPUT elements.  ``original_cell_index``
        # gives the input element index for each DOLFINx cell (serial run);
        # fall back to matching cell midpoints against element centroids.
        orig = getattr(domain.topology, "original_cell_index", None)
        if orig is not None and len(orig) >= n_cells:
            cell_to_elem = np.asarray(orig)[:n_cells]
        else:
            from scipy.spatial import cKDTree

            midpoints = dolfinx.mesh.compute_midpoints(
                domain, tdim, np.arange(n_cells, dtype=np.int32)
            )
            centroids = mesh.nodes[mesh.elements].mean(axis=1)
            if centroids.shape[1] == 2:
                centroids = np.column_stack(
                    [centroids, np.zeros(len(centroids))]
                )
            _, cell_to_elem = cKDTree(centroids).query(midpoints[:, :3])
            cell_to_elem = np.asarray(cell_to_elem)

        elem_stress = np.zeros((mesh.n_elements, 6))
        elem_stress[cell_to_elem] = voigt_stress

        # Average element stresses onto nodes using the INPUT connectivity so
        # the field is node-indexed like CalculiX's nodal stress output.
        nodal_stress = np.zeros((n_nodes, 6))
        counts = np.zeros(n_nodes)
        flat_nodes = mesh.elements.ravel()
        np.add.at(
            nodal_stress, flat_nodes,
            np.repeat(elem_stress, mesh.elements.shape[1], axis=0),
        )
        np.add.at(counts, flat_nodes, 1.0)
        counts[counts == 0.0] = 1.0
        nodal_stress /= counts[:, None]

        stress_field = StressField(values=nodal_stress, location="nodes")

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
        """Steady-state thermal solve using FEniCSx.

        Assembles the steady heat equation ``-∇·(k ∇T) = Q`` with
        Dirichlet, flux, and convection boundary conditions pulled
        from ``load_case``.

        Parameters
        ----------
        mesh, material, load_case
            See [SolverBackend.solve_thermal_steady][feaweld.solver.backend.SolverBackend.solve_thermal_steady].
        """
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

        # Dirichlet BCs (fixed temperature).  Node-set dofs are matched by
        # coordinate because DOLFINx reorders vertices relative to the input
        # mesh (the original node ids are not valid DOLFINx vertex indices).
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.TEMPERATURE:
                T_bc_val = constraint.values[0]
                T_bc = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_bc_val))
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = _block_dofs_for_nodes(V, mesh, node_ids)
                else:
                    domain.topology.create_connectivity(
                        domain.topology.dim - 1, domain.topology.dim
                    )
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

        # Reorder the scalar temperature field into input node order.
        block_dofs = _block_dofs_for_nodes(V, mesh, np.arange(mesh.n_nodes))
        result_temp = np.asarray(Th.x.array)[block_dofs].astype(np.float64)

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
    ) -> FEAResults:
        """Transient thermal solve using backward-Euler time stepping.

        Supports an optional moving heat source (e.g.
        [GoldakHeatSource][feaweld.solver.thermal.GoldakHeatSource]) that is
        evaluated at every node and every time step.

        Parameters
        ----------
        mesh, material, load_case, time_steps, heat_source
            See [SolverBackend.solve_thermal_transient][feaweld.solver.backend.SolverBackend.solve_thermal_transient].

        Returns
        -------
        FEAResults
            ``temperature`` is a ``(n_steps, n_nodes)`` array and
            ``time_steps`` echoes the input time vector.
        """
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

        # Dirichlet BCs.  Node-set dofs are matched by coordinate because
        # DOLFINx reorders vertices relative to the input mesh (the original
        # node ids are not valid DOLFINx vertex indices).
        bcs = []
        for constraint in load_case.constraints:
            if constraint.bc_type == LoadType.TEMPERATURE:
                T_bc_val = constraint.values[0]
                T_bc = dolfinx.fem.Constant(domain, PETSc.ScalarType(T_bc_val))
                if constraint.node_set in mesh.node_sets:
                    node_ids = mesh.node_sets[constraint.node_set]
                    dofs = _block_dofs_for_nodes(V, mesh, node_ids)
                else:
                    domain.topology.create_connectivity(
                        domain.topology.dim - 1, domain.topology.dim
                    )
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

        # Storage for temperature history, kept in input node order.  The
        # block dofs map each input node to its DOLFINx dof so every recorded
        # step is reindexed consistently.
        block_dofs = _block_dofs_for_nodes(V, mesh, np.arange(mesh.n_nodes))
        n_steps = len(time_steps)
        temp_history = np.zeros((n_steps, mesh.n_nodes))
        temp_history[0, :] = np.asarray(T_n.x.array)[block_dofs]

        # PETSc solver
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        T_sol = dolfinx.fem.Function(V)

        for step_idx in range(1, n_steps):
            dt_val = time_steps[step_idx] - time_steps[step_idx - 1]
            dt_const.value = dt_val
            t_current = time_steps[step_idx]

            # Update heat source if provided.  Evaluate at the temperature
            # space's dof coordinates (not domain.geometry.x, whose ordering
            # differs from the dof ordering) so Q is assigned dof-consistently.
            if heat_source is not None and hasattr(heat_source, "evaluate"):
                dof_coords = V.tabulate_dof_coordinates()
                q_vals = heat_source.evaluate(
                    dof_coords[:, 0], dof_coords[:, 1], dof_coords[:, 2], t_current
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

            temp_history[step_idx, :] = np.asarray(T_sol.x.array)[block_dofs]

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
        """Sequential thermomechanical coupling: thermal first, then mechanical.

        Delegates to [feaweld.solver.thermomechanical.sequential_coupled_solve][],
        which runs a transient thermal solve at each time step and
        uses the temperature field as a thermal-strain load on a
        mechanical solve.

        Parameters
        ----------
        mesh, material, mechanical_lc, thermal_lc, time_steps
            See [SolverBackend.solve_coupled][feaweld.solver.backend.SolverBackend.solve_coupled].
        """
        from feaweld.solver.thermomechanical import sequential_coupled_solve

        return sequential_coupled_solve(
            backend=self,
            mesh=mesh,
            material=material,
            thermal_lc=thermal_lc,
            mechanical_lc=mechanical_lc,
            time_steps=time_steps,
        )
