import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import date, datetime, timedelta
import geopandas as gpd
import seaborn as sns
import pandas as pd
import rioxarray
import pickle
import shapely
from shapely.geometry import mapping
import copy
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import h5py
import ast
from hmc_core import HMCSolver, FluxEdge, TimeStep
import modvis.ats_xdmf as xdmf
from modvis.ats_xdmf import meshXYZ

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================
case_name = sys.argv[1]
start = int(sys.argv[2])
stop = int(sys.argv[3])

# ============================================================
# Load pre-analyzed file
# ============================================================
surf_ids=np.arange(2000000, 2000000+92832)
surface_vis=xdmf.VisFile(case_name, model_time_unit='d',domain='surface', load_mesh=True, columnar=True, mixed_element=True)
subsurface_vis=xdmf.VisFile(case_name, model_time_unit='d', load_mesh=True, columnar=True, mixed_element=True)
map_flat_surf = np.asarray(surface_vis.map).reshape(-1)
map_flat_subsurf=np.asarray(subsurface_vis.map).reshape(-1)
df=pd.read_csv('data/intermediate_df.csv', index_col='Unnamed: 0')

# ============================================================
# SETTINGS
# ============================================================

h5_file = case_name+"/ats_vis_data.h5"
h5_file2 = case_name+"/ats_vis_surface_data.h5"

DT = 1.0          # days
MAX_CFL = 0.9


# ============================================================
# PREPARE DATAFRAME
# ============================================================

df = df.copy()


#df["neighbor_id"]

element_ids = df.index.to_numpy(dtype=int)

N = len(df)

element_to_position = {
    int(eid): i
    for i, eid in enumerate(element_ids)
}

# ============================================================
# READ ATS VARIABLE
# ============================================================

def read_timestep(f, variable, timestep,map_type):
    if variable=='velocity.z':
        try:    
            file=np.asarray(
                    f['darcy_'+variable][str(timestep)][:],
                    dtype=float
                ).reshape(-1)[map_type]
        except KeyError:    
            file=np.asarray(
                    f['surface-surface_subsurface_flux'][str(timestep)][:],
                    dtype=float
                ).reshape(-1)[map_type]
    elif variable=='precipitation_rain' or variable=='total_evapotranspiration':
        try:    
            file=np.asarray(
                    f['surface-'+variable][str(timestep)][:],
                    dtype=float
                ).reshape(-1)[map_type]
        except KeyError:    
            file=np.zeros(len(map_type))
    else:
        try:    
            file=np.asarray(
                    f[variable][str(timestep)][:],
                    dtype=float
                ).reshape(-1)[map_type]
        except KeyError:    
            try:    
                file=np.asarray(
                        f['surface-'+variable][str(timestep)][:],
                        dtype=float
                    ).reshape(-1)[map_type]
            except KeyError:    
                file=np.asarray(
                        f['darcy_'+variable][str(timestep)][:],
                        dtype=float
                    ).reshape(-1)[map_type]
        
    return file


# ============================================================
# INITIAL HMC SOURCE COMPOSITION
# ============================================================

sources = [
    "top",
    "geol",
    "bedrock",
    "surf",
    "precip",
]

initial_fractions = {
    source: np.zeros(N)
    for source in sources
}


for i, layer in enumerate(df["layer"]):

    layer = str(layer).strip().lower()

    if layer == "top":
        initial_fractions["top"][i] = 1.0

    elif layer == "geol":
        initial_fractions["geol"][i] = 1.0

    elif layer == "bedrock":
        initial_fractions["bedrock"][i] = 1.0

    elif layer == "surf":
        initial_fractions["surf"][i] = 1.0

    else:
        raise ValueError(
            f"Unknown layer '{layer}' "
            f"for element {element_ids[i]}"
        )


# ============================================================
# HMC SOLVER
# ============================================================

solver = HMCSolver(
    n_cells=N,
    sources=sources,
    initial_fractions=initial_fractions,
    max_cfl=MAX_CFL,
)


# ============================================================
# BUILD INTERCELL FLUXES
# ============================================================

# ------------------------------------------------------------
# Positions of the surface cells, in the SAME order as surf_ids /
# map_flat_surf, so ff[t_new][map_flat_surf][k] always lands on
# surf_pos[k]. Computed once -- no more df.loc[surf_ids, ...]
# label-based lookups anywhere near the timestep loop.
# ------------------------------------------------------------
surf_pos = np.array([element_to_position[eid] for eid in surf_ids], dtype=int)
is_surf = np.zeros(N, dtype=bool)
is_surf[surf_pos] = True

surf_area_arr = df["surf_area"].to_numpy(dtype=float)
cell_thickness_static = df["cell_thickness"].to_numpy(dtype=float)

width = np.sqrt(surf_area_arr)

# ------------------------------------------------------------
# Static edge topology + orientation -- pure geometry, computed
# once, never touched again regardless of how many timesteps run.
# ------------------------------------------------------------
def precompute_static_geometry(df, element_to_position, width):
    centroid = df[["x", "y", "z"]].to_numpy(dtype=float)

    pairs_i, pairs_j, seen = [], [], set()
    for eid, row in df.iterrows():
        i = element_to_position[eid]
        for nid in ast.literal_eval(row["neighbor_id"]):
            if nid not in element_to_position:
                print('ID not found; double check')
                continue
            j = element_to_position[nid]
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            pairs_i.append(key[0]); pairs_j.append(key[1])

    i_arr = np.asarray(pairs_i, dtype=int)
    j_arr = np.asarray(pairs_j, dtype=int)

    d = centroid[j_arr] - centroid[i_arr]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    is_vertical = np.abs(dz) > np.maximum(np.abs(dx), np.abs(dy))

    horiz_dist = np.sqrt(dx**2 + dy**2)
    safe_horiz = np.where(horiz_dist > 0, horiz_dist, 1.0)  # placeholder; result discarded anyway when is_vertical
    nx = np.where(~is_vertical, dx / safe_horiz, 0.0)
    ny = np.where(~is_vertical, dy / safe_horiz, 0.0)
    vert_sign = np.where(is_vertical, np.sign(dz), 0.0)

    return (i_arr, j_arr, is_vertical, nx, ny, vert_sign,
            width[i_arr], width[j_arr])

i_arr, j_arr, is_vertical, nx, ny, vert_sign, width_i, width_j = \
    precompute_static_geometry(df, element_to_position, width)

# Vertical face area depends only on (static) surf_area -- never
# recomputed inside the loop.
face_area_vert_static = 0.5 * (surf_area_arr[i_arr] + surf_area_arr[j_arr])

# The ONE array that actually changes every timestep.
cell_h_current = cell_thickness_static.copy()


def build_flux_edges_vectorized(cell_h_current, vx, vy, vz, dt):
    h_i, h_j = cell_h_current[i_arr], cell_h_current[j_arr]
    face_area_horiz = 0.5 * (width_i + width_j) * 0.5 * (h_i + h_j)

    q_vert = 0.5 * (vz[i_arr] + vz[j_arr])
    volume_vert = q_vert * face_area_vert_static * dt * vert_sign

    q_i = vx[i_arr] * nx + vy[i_arr] * ny
    q_j = vx[j_arr] * nx + vy[j_arr] * ny
    volume_horiz = 0.5 * (q_i + q_j) * face_area_horiz * dt

    signed_volume = np.where(is_vertical, volume_vert, volume_horiz)
    src = np.where(signed_volume > 0, i_arr, j_arr)
    dst = np.where(signed_volume > 0, j_arr, i_arr)
    transfer_volume = np.abs(signed_volume)
    keep = transfer_volume > 0

    return src[keep], dst[keep], transfer_volume[keep]


# ============================================================
# GET ATS TIMESTEPS
# ============================================================

with h5py.File(h5_file, "r") as f:

    timesteps = sorted(
        int(k)
        for k in f["water_content"].keys()
    )

# ============================================================
# RUN HMC
# ============================================================

results = []


with h5py.File(h5_file, "r") as f:
    with h5py.File(h5_file2, "r") as ff:
        for k in range(start, stop):

            t_old = timesteps[k - 1]
            t_new = timesteps[k]

            print(
                f"Processing "
                f"{k}/{len(timesteps)-1} "
                f"(timestep {t_new})"
            )
            #df.loc[surf_ids,'cell_thickness']=np.asarray(ff['surface-ponded_depth/'+str(t_new)][:][map_flat_surf]).reshape(-1, )
            cell_h_current[surf_pos] = np.asarray(ff['surface-ponded_depth/'+str(t_new)][:][map_flat_surf]).reshape(-1, )
            #inactive_mask  = is_surf & (cell_h_current < 0.02)
            # ====================================================
            # WATER STORAGE
            # ====================================================

            v_old1 = read_timestep(
                f,
                "water_content",
                t_old, map_flat_subsurf
            )
            v_old2 = read_timestep(
                ff,
                "water_content",
                t_old, map_flat_surf
            )
            v_old=np.hstack((v_old1,v_old2))/55000#used to convert mol of water to vol

            v_new1 = read_timestep(
                f,
                "water_content",
                t_new, map_flat_subsurf
            )
            v_new2 = read_timestep(
                ff,
                "water_content",
                t_new, map_flat_surf
            )
            v_new=np.hstack((v_new1,v_new2))/55000#used to convert mol of water to vol

            # ====================================================
            # DARCY VELOCITY
            # ====================================================

            vx1 = read_timestep(
                f,
                "velocity.x",
                t_new, map_flat_subsurf
            )
            vx2 = read_timestep(
                ff,
                "velocity.x",
                t_new, map_flat_surf
            )
            vx=np.hstack((vx1,vx2))

            vy1 = read_timestep(
                f,
                "velocity.y",
                t_new, map_flat_subsurf
            )
            vy2 = read_timestep(
                ff,
                "velocity.y",
                t_new, map_flat_surf
            )
            vy=np.hstack((vy1,vy2))

            vz1 = read_timestep(
                f,
                "velocity.z",
                t_new, map_flat_subsurf
            )
            vz2 = read_timestep(
                ff,
                "velocity.z",
                t_new, map_flat_surf
            )
            vz=np.hstack((vz1,vz2))
            
            # ====================================================
            # INTERCELL FLUX
            # ====================================================

            src_arr, dst_arr, vol_arr = build_flux_edges_vectorized(cell_h_current, vx, vy, vz, DT)
            # ====================================================
            # PRECIPITATION
            # ====================================================

            precip1 = read_timestep(
                f,
                "precipitation_rain",
                t_new, map_flat_subsurf
            )
            precip2 = read_timestep(
                ff,
                "precipitation_rain",
                t_new, map_flat_surf
            )
            precip=np.hstack((precip1,precip2))

            precip_volume = (
                precip
                * df["surf_area"].to_numpy(dtype=float)
                * DT
            )

            # ====================================================
            # EVAPOTRANSPIRATION
            # ====================================================

            ET1 = read_timestep(
                f,
                "total_evapotranspiration",
                t_new, map_flat_subsurf
            )
            ET2 = read_timestep(
                ff,
                "total_evapotranspiration",
                t_new, map_flat_surf
            )
            ET=np.hstack((ET1,ET2))

            ET_volume = (
                ET
                * df["surf_area"].to_numpy(dtype=float)
                * DT
            )

            # ====================================================
            # HMC TIMESTEP
            # ====================================================

            ts = TimeStep(
                v_old=v_old,
                v_new=v_new,
                edge_src=src_arr, edge_dst=dst_arr, edge_volume=vol_arr,
                bc_out=ET_volume,
                bc_in={
                    "precip": precip_volume
                },dt=DT,
            )

            report = solver.step(ts)

            # ====================================================
            # STORE FRACTIONS
            # ====================================================

            #as a fixed-column-order 2D array:
            fractions = np.stack([solver.as_array(w) for w in sources], axis=1)
            #zero_out_surf_source_when_thin(solver, is_surf, cell_h_current, threshold=0.02, source_name="surf")

            results.append({
                "timestep": t_new,
                "fractions": fractions,
            })
with open(case_name+"_HMC_result.pkl", "wb") as f:
    pickle.dump(results, f)