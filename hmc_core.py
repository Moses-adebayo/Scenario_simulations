"""
Hydraulic Mixing-Cell (HMC) method
===================================

A clean-room implementation of the Hydraulic Mixing-Cell (HMC) algorithm
introduced by:

    Partington, D., Brunner, P., Simmons, C. T., Therrien, R., Werner, A. D.,
    Dandy, G. C., & Maier, H. R. (2011). A hydraulic mixing-cell method to
    quantify the groundwater component of streamflow within spatially
    distributed fully integrated surface water-groundwater flow models.
    Environmental Modelling & Software, 26(7), 886-898.
    https://doi.org/10.1016/j.envsoft.2011.02.007

and extended (sub-time-stepping for stability) in:

    Partington, D., Brunner, P., Simmons, C. T., Werner, A. D., & Therrien, R.
    (2013). Reactive Interpreting streamflow generation mechanisms from
    integrated surface-subsurface flow models of a riparian wetland and
    catchment. Water Resources Research, 49, 5501-5519.
    https://doi.org/10.1002/wrcr.20356

as used e.g. by:

    Glaser, B., Hopp, L., Partington, D., Brunner, P., Therrien, R., & Klaus, J.
    (2021). Sources of surface water in space and time: Identification of
    delivery processes and geographical sources with hydraulic mixing-cell
    modeling. Water Resources Research, 57, e2021WR030332.
    https://doi.org/10.1029/2021WR030332

WHAT THE METHOD DOES
---------------------
HMC is a *post-processing* routine applied on top of an already-computed,
transient, spatially distributed flow solution (e.g. from HydroGeoSphere,
ParFlow, MODFLOW, ...). It does not require a separate solute-transport
simulation. Every model cell / control volume is treated as instantaneously
and perfectly mixed. The water occupying each cell is partitioned into a set
of user-defined "sources" (e.g. stream water, groundwater inflow, rainfall
recharge, an arbitrary initial condition). Outflow from a cell carries the
cell's own (donor / upwind) composition; inflow from a neighbouring cell
carries that neighbour's previous-time-step composition; boundary-condition
inflows are tagged a priori with the source they represent.

THE UPDATE EQUATION
--------------------
Following Partington et al. (2011, their Eq. 1; reproduced identically in
Nogueira et al., 2022, HESS, Eq. 1), for cell i and source w:

    f_i(w)^t = f_i(w)^(t-1) * [ V_i^(t-1) - Vbc_out_i^t - sum_j V_(i->j) ] / V_i^t
             + [ Vbc_in_i(w)^t + sum_j V_(j->i) * f_j(w)^(t-1) ] / V_i^t

Rearranged as a mass balance (multiply through by V_i^t):

    (new mass of source w in cell i)
        = f_i(w)^(t-1) * (volume of "old" water that stays in cell i)
        + Vbc_in_i(w)^t                          (boundary inflow of source w)
        + sum_j V_(j->i) * f_j(w)^(t-1)           (inflow from neighbours)

Summed over all sources w, this collapses to the ordinary water-volume mass
balance of the cell. Consequently, if the underlying flow solution is itself
volume-conservative, then sum_w f_i(w)^t == 1 exactly (to round-off) for
every cell and every time step -- this is the standard internal consistency
/ "relative error" check used throughout the HMC literature (cf. Partington
et al., 2011, their Eq. 7).

STABILITY: HMC SUB-TIME-STEPPING
----------------------------------
The update above is only physically valid if the volume leaving a cell
within a step does not exceed the volume that was in the cell at the start
of the step -- otherwise the coefficient multiplying f_i(w)^(t-1) goes
negative, which is unstable and unphysical (Partington et al., 2013).
Because flow-model time steps are often far larger than this "Courant-type"
limit requires, HMC uses an independent internal sub-time-stepping scheme:
a single flow time step is subdivided into as many equal HMC sub-steps as
needed to keep, in every cell, the fractional outflow over a sub-step below
a threshold (`max_cfl`, default 0.9). Volumes and fluxes are assumed to vary
linearly across the flow time step and are interpolated accordingly for the
sub-steps.

CAVEAT
------
This module was written directly from the published method description; it
is NOT the original code of Partington, Brunner, Therrien and co-authors,
which (as far as could be established) has never been publicly released.
Always check the mass-balance diagnostics in `MassBalanceReport` before
trusting results for a real study, and see `verify_hmc.py` for tests against
an analytical solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Data structures describing one flow time step
# --------------------------------------------------------------------------- #

@dataclass
class FluxEdge:
    """A directed volumetric flux between two cells over one flow time step.

    Attributes
    ----------
    src, dst : int
        Index of the upstream (src) and downstream (dst) cell.
    volume : float
        Non-negative volume of water transferred from `src` to `dst` over
        the *whole* flow time step [L^3]. If flow between the same pair of
        cells reverses direction at different points within the time step,
        represent it with two separate edges (src->dst and dst->src); HMC
        (like the flow model itself) works with the net exchanged volumes
        that the flow solver reports for each direction/face.
    """

    src: int
    dst: int
    volume: float


@dataclass
class TimeStep:
    """Everything HMC needs to advance the fractions by one flow time step.

    Attributes
    ----------
    v_old, v_new : (n_cells,) array
        Cell water volumes (or, for variably saturated models, the moisture
        volume: cell volume x saturation x porosity) at the start (t-1) and
        end (t) of the step [L^3].
    bc_out : (n_cells,) array
        Total volume leaving each cell through *any* outflow boundary
        condition during the step (e.g. discharge to a stream outlet,
        evapotranspiration, pumping). Its composition is whatever is
        currently in the cell, so it does not need to be tagged by source.
    edges : list of FluxEdge, optional
        Inter-cell fluxes during the step, as a plain Python list -- fine
        for small meshes or quick scripts. At watershed scale (thousands of
        edges), building a Python object per edge and having `step()`
        iterate them is real, measurable overhead; prefer `edge_src` /
        `edge_dst` / `edge_volume` below instead, which is what `step()`
        converts this into internally anyway. Ignored if those are given.
    edge_src, edge_dst, edge_volume : 1-D array, optional
        The same information as `edges`, as flat arrays (one entry per
        edge): `edge_src[k] -> edge_dst[k]` carries `edge_volume[k]` this
        step. This is the fast path -- if your edge-building code already
        produces numpy arrays (e.g. from a vectorized face-flux
        calculation), pass them directly here instead of wrapping each one
        in a `FluxEdge` only to have `step()` unwrap it again. Either
        supply all three, or none (and use `edges` instead).
    bc_in : dict[str, (n_cells,) array]
        Volume entering each cell through an inflow boundary condition
        during the step, tagged by the source name it represents (e.g.
        {"stream": ..., "groundwater": ..., "rainfall": ...}). Only
        sources actually present in `HMCSolver.sources` are used.
    dt : float
        Duration of the step [T]. Only used for bookkeeping / optional
        linear interpolation; the update itself is volume-, not flux-,
        based.
    """

    v_old: Array
    v_new: Array
    bc_out: Array
    edges: List[FluxEdge] = field(default_factory=list)
    edge_src: Array | None = None
    edge_dst: Array | None = None
    edge_volume: Array | None = None
    bc_in: Dict[str, Array] = field(default_factory=dict)
    dt: float = 1.0


@dataclass
class MassBalanceReport:
    """Diagnostics for one completed HMC step (see Partington et al., 2011, Eq. 7)."""

    max_fraction_sum_error: float           # max_i | sum_w f_i(w) - 1 |, ALL cells
    max_fraction_sum_error_wet: float       # same, but excluding cells that end this step "dry"
    max_negative_fraction: float            # most negative fraction found (should be ~0)
    n_substeps: int                         # number of HMC sub-steps used
    max_cfl_before_substepping: float       # worst-case fractional outflow before splitting
    n_dry_cells: int                        # cells with v_new <= dry_tol at the end of this step
    hit_substep_cap: bool                   # True if `max_substeps` was reached without
                                             # satisfying `max_cfl` for every non-dry cell
                                             # (this is EXPECTED and harmless if it only ever
                                             # happens together with dry cells; if it happens
                                             # for cells that are NOT ending the step dry, your
                                             # results for those cells may be under-resolved --
                                             # see the module docstring / README)


# --------------------------------------------------------------------------- #
# The HMC solver
# --------------------------------------------------------------------------- #

class HMCSolver:
    """Tracks per-cell source fractions through a sequence of flow time steps.

    Parameters
    ----------
    n_cells : int
        Number of control volumes (mesh cells / nodes / elements) tracked.
    sources : sequence of str
        Names of the water sources to track (e.g. "stream", "groundwater",
        "rainfall", "initial"). Fractions are stored in this order.
    initial_fractions : dict[str, array], optional
        Initial condition f_i(w)^0 for each source, shape (n_cells,). Any
        source not given is assumed to be zero everywhere. If omitted
        entirely, everything is assigned to a source literally named
        "initial" if present in `sources`, else left at zero (not
        recommended -- see `HMCSolver.set_uniform_initial_source`).
    max_cfl : float, default 0.9
        Maximum fraction of a cell's water allowed to leave in a single HMC
        sub-step before the step is subdivided further (Partington et al.,
        2013). Must be in (0, 1).
    """

    def __init__(
        self,
        n_cells: int,
        sources: Sequence[str],
        initial_fractions: Dict[str, Array] | None = None,
        max_cfl: float = 0.9,
        dry_tol: float | None = None,
        max_substeps: int = 256,
    ) -> None:
        if not (0.0 < max_cfl < 1.0):
            raise ValueError("max_cfl must be in (0, 1)")
        self.n_cells = n_cells
        self.sources: List[str] = list(sources)
        self.max_cfl = max_cfl
        # `dry_tol`: absolute volume below which a cell is considered "dry"
        # for stability-check purposes (see `step()` for why this matters).
        # If None, it is set automatically, per step, to a small fraction of
        # the largest cell volume seen in that step -- override with an
        # explicit value if your volumes are in unusual units or your
        # smallest physically-meaningful volume differs a lot from that
        # heuristic.
        self.dry_tol = dry_tol
        self.max_substeps = max_substeps

        self.fractions: Dict[str, Array] = {
            w: np.zeros(n_cells, dtype=float) for w in self.sources
        }
        if initial_fractions:
            for w, arr in initial_fractions.items():
                if w not in self.fractions:
                    raise KeyError(f"Unknown source '{w}' in initial_fractions")
                self.fractions[w] = np.asarray(arr, dtype=float).copy()

        self.history: List[Dict[str, Array]] = [self._snapshot()]
        self.reports: List[MassBalanceReport] = []

    # -- convenience setup -------------------------------------------------- #

    def set_uniform_initial_source(self, source: str) -> None:
        """Set every cell to 100% of `source` at t=0 (a common HMC spin-up trick:
        run a warm-up period with an "initial" source until it is flushed out,
        then discard that period -- see Glaser et al. 2021 and Nogueira et al.
        2022, who both do this)."""
        if source not in self.fractions:
            raise KeyError(source)
        for w in self.fractions:
            self.fractions[w][:] = 0.0
        self.fractions[source][:] = 1.0
        self.history[0] = self._snapshot()

    def _snapshot(self) -> Dict[str, Array]:
        return {w: arr.copy() for w, arr in self.fractions.items()}

    # -- core update ---------------------------------------------------------#

    def step(self, ts: TimeStep) -> MassBalanceReport:
        """Advance the fractions by one flow time step, using as many internal
        HMC sub-steps as required for stability, and record diagnostics."""

        n = self.n_cells
        v_old, v_new = np.asarray(ts.v_old, float), np.asarray(ts.v_new, float)
        bc_out = np.asarray(ts.bc_out, float)

        # Edges are converted to flat numpy arrays exactly ONCE per step
        # here, regardless of how many HMC sub-steps end up being needed --
        # this used to be redone (rebuilding a whole list of FluxEdge
        # Python objects) on every single sub-step, which dominated runtime
        # at watershed scale (thousands of edges x tens-to-hundreds of
        # sub-steps). If you already have flat arrays (e.g. from a
        # vectorized face-flux calculation), pass them via
        # `TimeStep.edge_src/edge_dst/edge_volume` directly and skip the
        # FluxEdge-object round trip entirely.
        if ts.edge_src is not None:
            edge_src = np.asarray(ts.edge_src, dtype=np.int64)
            edge_dst = np.asarray(ts.edge_dst, dtype=np.int64)
            edge_vol = np.asarray(ts.edge_volume, dtype=np.float64)
        elif ts.edges:
            n_edges = len(ts.edges)
            edge_src = np.fromiter((e.src for e in ts.edges), dtype=np.int64, count=n_edges)
            edge_dst = np.fromiter((e.dst for e in ts.edges), dtype=np.int64, count=n_edges)
            edge_vol = np.fromiter((e.volume for e in ts.edges), dtype=np.float64, count=n_edges)
        else:
            edge_src = np.empty(0, dtype=np.int64)
            edge_dst = np.empty(0, dtype=np.int64)
            edge_vol = np.empty(0, dtype=np.float64)

        # total outflow (to boundary + to neighbouring cells) per source cell,
        # assumed uniform in time across the flow time step. `np.bincount`
        # is a single compiled scatter-add over all edges at once, replacing
        # what used to be a Python `for e in edges: ...` loop.
        internal_out = np.bincount(edge_src, weights=edge_vol, minlength=n)
        total_out = bc_out + internal_out

        # A cell whose volume at the *end* of this macro step is (numerically)
        # zero is a special case: if it drains at a uniform rate all the way
        # to exactly empty, the ratio of outflow to remaining storage in the
        # very last instant of the step is mathematically unbounded no
        # matter how finely the step is subdivided (the limiting sub-step
        # ratio converges to total_out/v_old, a fixed number that does not
        # shrink as n_sub grows -- see README/notes for the derivation). No
        # finite number of sub-steps can satisfy an ordinary CFL target for
        # such a cell, so chasing one would either loop forever or (as an
        # earlier, buggy version of this code did) burn tens of thousands of
        # iterations for no benefit. Since the cell ends the step with ~0
        # water regardless of the path taken to get there, we simply exclude
        # it from the stability requirement -- its fractions are zeroed out
        # by the ordinary "v_end <= dry_tol" handling in `_substep` anyway.
        dry_tol = self.dry_tol
        if dry_tol is None:
            scale = float(max(np.max(v_old) if n else 0.0, np.max(v_new) if n else 0.0, 1.0))
            dry_tol = 1e-8 * scale
        ends_dry = v_new <= dry_tol

        def worst_local_cfl(n_sub: int) -> float:
            """Worst-case fraction of a cell's *sub-step-starting* volume
            that would leave within a single HMC sub-step, checked at every
            sub-step boundary (not just t=0), for cells that do NOT end this
            macro step dry. This matters whenever a cell's storage is
            shrinking over the flow time step: the available "cushion" at
            the *end* of the step can be much smaller than at the start,
            even though the same total outflow is (by assumption) spread
            evenly across the step."""
            worst = 0.0
            local_out = total_out / n_sub
            for k in range(n_sub):
                v_start_k = v_old + (v_new - v_old) * (k / n_sub)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(
                        (~ends_dry) & (v_start_k > 1e-12),
                        local_out / np.maximum(v_start_k, 1e-300),
                        0.0,
                    )
                worst = max(worst, float(np.max(ratio)) if n else 0.0)
                if not np.isfinite(worst):
                    return worst
            return worst

        # Cheap initial guess based on the worse of the two step endpoints,
        # then refine by actually checking every sub-step boundary (doubling
        # n_sub until the check passes, capped to avoid runaway loops on
        # pathological/inconsistent input -- see `ends_dry` above for the
        # one case this cap is actually expected to bind on).
        ref_vol = np.where(ends_dry, np.inf, np.minimum(v_old, v_new))
        with np.errstate(divide="ignore", invalid="ignore"):
            cfl_guess = np.where(
                np.isfinite(ref_vol) & (ref_vol > 1e-12),
                total_out / np.maximum(ref_vol, 1e-300),
                0.0,
            )
        finite_guess = cfl_guess[np.isfinite(cfl_guess)]
        max_cfl_full_step = float(np.max(finite_guess)) if finite_guess.size else 0.0

        n_sub = max(1, int(np.ceil(max_cfl_full_step / self.max_cfl))) if max_cfl_full_step > 0 else 1
        n_sub = min(n_sub, self.max_substeps)
        _MAX_DOUBLINGS = 20
        hit_cap = False
        for _ in range(_MAX_DOUBLINGS):
            if worst_local_cfl(n_sub) <= self.max_cfl:
                break
            if n_sub >= self.max_substeps:
                hit_cap = True
                break
            n_sub = min(n_sub * 2, self.max_substeps)

        # Linear interpolation of volumes/fluxes across sub-steps. Edge
        # volumes are scaled by dividing the flat array once (cheap) rather
        # than rebuilding FluxEdge objects every sub-step (the other big
        # cost this used to have).
        v0 = v_old
        for k in range(1, n_sub + 1):
            frac_end = k / n_sub
            frac_start = (k - 1) / n_sub
            v_start = v_old + (v_new - v_old) * frac_start
            v_end = v_old + (v_new - v_old) * frac_end

            sub_edge_vol = edge_vol / n_sub if n_sub != 1 else edge_vol
            sub_bc_out = bc_out / n_sub
            sub_bc_in = {w: arr / n_sub for w, arr in ts.bc_in.items()}

            self._substep(v_start, v_end, edge_src, edge_dst, sub_edge_vol, sub_bc_out, sub_bc_in, dry_tol)
            v0 = v_end  # noqa: F841  (kept for clarity / potential debugging)

        # -- diagnostics --------------------------------------------------- #
        total = np.zeros(n)
        min_frac = 0.0
        for w in self.sources:
            total += self.fractions[w]
            min_frac = min(min_frac, float(np.min(self.fractions[w])))
        max_err = float(np.max(np.abs(total - 1.0))) if n else 0.0
        wet_mask = ~ends_dry
        max_err_wet = (
            float(np.max(np.abs(total[wet_mask] - 1.0))) if n and np.any(wet_mask) else 0.0
        )

        report = MassBalanceReport(
            max_fraction_sum_error=max_err,
            max_fraction_sum_error_wet=max_err_wet,
            max_negative_fraction=min_frac,
            n_substeps=n_sub,
            max_cfl_before_substepping=max_cfl_full_step,
            n_dry_cells=int(np.sum(ends_dry)),
            hit_substep_cap=hit_cap,
        )
        self.reports.append(report)
        self.history.append(self._snapshot())
        return report

    def _substep(
        self,
        v_start: Array,
        v_end: Array,
        edges: List[FluxEdge],
        bc_out: Array,
        bc_in: Dict[str, Array],
        dry_tol: float = 0.0,
    ) -> None:
        """One (already CFL-safe) HMC sub-step: implements the Partington et
        al. (2011) Eq. 1 update for every tracked source simultaneously.

        Composition is defined as new_mass(w) / total_mass, where
        total_mass = retained_volume + total_inflow (the total water that
        was present or arrived in the cell this sub-step, regardless of how
        much of it is still stored at the end). Whenever the sub-step
        respects the CFL condition (the normal case), total_mass is
        algebraically identical to v_end, so this is exactly the textbook
        HMC update. It only differs -- deliberately -- in two edge cases,
        both of which matter for real applications and are handled here
        rather than left as bugs:

        1. A cell whose *outflow* this sub-step exceeds its starting volume
           (only possible right at the edges of a CFL-violating step, e.g.
           a cell that starts the step exactly empty). Some of that outflow
           must have been sourced from the very inflow arriving this
           sub-step; total_mass > v_end here, and dividing by total_mass
           (rather than v_end) is what keeps every fraction inside [0, 1]
           instead of exceeding 1.
        2. A "pass-through" cell with ~zero storage at the end of the
           sub-step but substantial simultaneous inflow and outflow (e.g. a
           zero-storage stream/channel node that simply routes whatever
           arrives straight back out every step). Dividing by v_end (~0)
           would either blow up or, with the old dry-cell convention, get
           unconditionally zeroed out -- discarding exactly the composition
           you need if you're reporting the makeup of that outflow (e.g. a
           streamflow hydrograph separation). Dividing by total_mass
           instead gives the correct flow-weighted composition of whatever
           is passing through, and only cells with NO significant water
           present OR flowing through them this sub-step (total_mass <=
           dry_tol) fall back to a fraction of 0 (composition is genuinely
           undefined there).

        There is deliberately no "force this cell dry" override here (an
        earlier version of this module had one). Zeroing a cell's
        composition wholesale also zeroes what it hands to its neighbours
        via `edges` -- any real outflow from a "forced dry" cell then
        arrives downstream credited to no source at all, silently breaking
        mass conservation beyond that one cell. If you need to suppress or
        override one *specific* source's contribution in specific cells
        (e.g. "this source shouldn't count once a cell gets this thin"),
        do it as a post-processing step on `HMCSolver.fractions` after
        calling `step()`, renormalising the remaining sources yourself --
        that only touches reporting, not the mixing/mass-balance physics
        computed here. See the worked example in
        `example_layered_streamflow.py` / the project README."""

    def _substep(
        self,
        v_start: Array,
        v_end: Array,
        edge_src: Array,
        edge_dst: Array,
        edge_vol: Array,
        bc_out: Array,
        bc_in: Dict[str, Array],
        dry_tol: float = 0.0,
    ) -> None:
        """One (already CFL-safe) HMC sub-step: implements the Partington et
        al. (2011) Eq. 1 update for every tracked source simultaneously.

        Composition is defined as new_mass(w) / total_mass, where
        total_mass = retained_volume + total_inflow (the total water that
        was present or arrived in the cell this sub-step, regardless of how
        much of it is still stored at the end). Whenever the sub-step
        respects the CFL condition (the normal case), total_mass is
        algebraically identical to v_end, so this is exactly the textbook
        HMC update. It only differs -- deliberately -- in two edge cases,
        both of which matter for real applications and are handled here
        rather than left as bugs:

        1. A cell whose *outflow* this sub-step exceeds its starting volume
           (only possible right at the edges of a CFL-violating step, e.g.
           a cell that starts the step exactly empty). Some of that outflow
           must have been sourced from the very inflow arriving this
           sub-step; total_mass > v_end here, and dividing by total_mass
           (rather than v_end) is what keeps every fraction inside [0, 1]
           instead of exceeding 1.
        2. A "pass-through" cell with ~zero storage at the end of the
           sub-step but substantial simultaneous inflow and outflow (e.g. a
           zero-storage stream/channel node that simply routes whatever
           arrives straight back out every step). Dividing by v_end (~0)
           would either blow up or, with the old dry-cell convention, get
           unconditionally zeroed out -- discarding exactly the composition
           you need if you're reporting the makeup of that outflow (e.g. a
           streamflow hydrograph separation). Dividing by total_mass
           instead gives the correct flow-weighted composition of whatever
           is passing through, and only cells with NO significant water
           present OR flowing through them this sub-step (total_mass <=
           dry_tol) fall back to a fraction of 0 (composition is genuinely
           undefined there).

        There is deliberately no "force this cell dry" override here (an
        earlier version of this module had one). Zeroing a cell's
        composition wholesale also zeroes what it hands to its neighbours
        via the edges -- any real outflow from a "forced dry" cell then
        arrives downstream credited to no source at all, silently breaking
        mass conservation beyond that one cell. If you need to suppress or
        override one *specific* source's contribution in specific cells
        (e.g. "this source shouldn't count once a cell gets this thin"),
        do it as a post-processing step on `HMCSolver.fractions` after
        calling `step()`, renormalising the remaining sources yourself --
        that only touches reporting, not the mixing/mass-balance physics
        computed here. See the worked example in
        `example_layered_streamflow.py` / the project README.

        Performance note: `edge_src`/`edge_dst`/`edge_vol` are flat numpy
        arrays (one entry per edge), not a list of `FluxEdge` objects.
        Every quantity below is built with `np.bincount` -- a single
        compiled scatter-add over all edges at once -- rather than a
        Python-level `for edge in edges: ...` loop with an inner loop over
        sources. That inner loop used to dominate runtime at watershed
        scale (it's O(edges x sources) *Python bytecode*, re-run on every
        sub-step); this is now O(sources) calls into compiled numpy code,
        each doing O(edges) work in C."""

        n = self.n_cells
        old_fractions = {w: arr.copy() for w, arr in self.fractions.items()}

        internal_out = np.bincount(edge_src, weights=edge_vol, minlength=n)
        outflow_wanted = bc_out + internal_out
        retained_volume = np.maximum(v_start - outflow_wanted, 0.0)

        total_inflow = np.zeros(n)
        internal_in_by_source: Dict[str, Array] = {}
        for w in self.sources:
            # Gather each edge's SOURCE-cell composition (a fast, vectorized
            # fancy-index gather), weight by that edge's volume, then
            # scatter-add into the destination cells -- the vectorized
            # equivalent of "for edge in edges: dst_mass[edge.dst] +=
            # edge.volume * old_fractions[w][edge.src]".
            weighted = edge_vol * old_fractions[w][edge_src]
            internal_in_by_source[w] = np.bincount(edge_dst, weights=weighted, minlength=n)
            total_inflow += bc_in.get(w, np.zeros(n)) + internal_in_by_source[w]

        total_mass = retained_volume + total_inflow  # == v_end whenever CFL holds

        has_composition = total_mass > dry_tol
        safe_total_mass = np.where(has_composition, total_mass, 1.0)

        for w in self.sources:
            inflow_bc = bc_in.get(w, np.zeros(n))
            new_mass = (
                old_fractions[w] * retained_volume
                + inflow_bc
                + internal_in_by_source[w]
            )
            self.fractions[w] = np.where(has_composition, new_mass / safe_total_mass, 0.0)

    # -- convenience access --------------------------------------------------#

    def as_array(self, source: str) -> Array:
        return self.fractions[source]

    def total_fraction(self) -> Array:
        total = np.zeros(self.n_cells)
        for w in self.sources:
            total += self.fractions[w]
        return total