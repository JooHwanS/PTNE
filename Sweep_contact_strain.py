# sweep_contact_vs_strain_materialwise_rigid_ref.py
# 10개 이미지 × (PDMS/Parylene/PI/Rigid 각각 고정 Rmin, valley sweep)
# strain = (electrode_length - L_elec_baseline) / L_elec_baseline * 100
# baseline: material_id="Rigid", valley=1000 일 때의 electrode_length (이미지별)

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

# ---------------- 공통 파라미터 ----------------
PIXELS_PER_UM  = 2.4
GAP_TOL_UM     = 2.0
L_LOCAL_UM     = 3.0
HYST_RATIO     = 1.2
BRIDGE_POST_UM = 1.0

IMAGES = [f"line{i}.tif" for i in range(1, 11)]

# 재료별 스윕 설정
MATERIALS = [
    {"id": "PDMS",    "Rmin_um": 10.0,   "valley_range": range(1,   51)},    # 1~50
    {"id": "Parylene","Rmin_um": 50.0,   "valley_range": range(30,  81)},    # 30~80
    {"id": "PI",      "Rmin_um": 200.0,  "valley_range": range(100, 201)},   # 100~200
    {"id": "Rigid",   "Rmin_um": 1000.0, "valley_range": range(1000,1051)},  # 1000~1050
]

OUT_DIR = "out_sweep3"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "per_image"), exist_ok=True)
px_um = 1.0/PIXELS_PER_UM

# ---------- 스켈레톤→경로→재표본화 ----------
def load_binary(path):
    arr = np.array(Image.open(path).convert("L")).astype(np.uint8)
    if not (np.unique(arr).size <= 2 and set(np.unique(arr)).issubset({0,255})):
        t = threshold_otsu(arr); bw = (arr >= t).astype(np.uint8)
    else:
        bw = (arr > 0).astype(np.uint8)
    return skeletonize(bw > 0).astype(np.uint8)

def neighbors8(y,x,H,W):
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W: yield ny, nx

def build_graph(sk):
    H,W = sk.shape
    nodes = [(y,x) for y,x in zip(*np.nonzero(sk))]
    idx = {(y,x):i for i,(y,x) in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for i,(y,x) in enumerate(nodes):
        for ny,nx in neighbors8(y,x,H,W):
            if sk[ny,nx]:
                adj[i].append(idx[(ny,nx)])
    return nodes, adj

def bfs_longest_path_from(start_idx, adj):
    N=len(adj); dist=[-1]*N; prev=[-1]*N
    q=deque([start_idx]); dist[start_idx]=0
    while q:
        u=q.popleft()
        for v in adj[u]:
            if dist[v]==-1:
                dist[v]=dist[u]+1; prev[v]=u; q.append(v)
    far=int(np.argmax(dist)); return dist,prev,far

def longest_geodesic_path(sk):
    nodes, adj = build_graph(sk)
    if not nodes: return None
    deg=[len(v) for v in adj]
    endpoints=[i for i,d in enumerate(deg) if d==1] or [0]
    _,_,farA=bfs_longest_path_from(endpoints[0], adj)
    _,prev2,farB=bfs_longest_path_from(farA, adj)
    path=[]; cur=farB
    while cur!=-1:
        path.append(cur)
        if cur==farA: break
        cur=prev2[cur]
    return np.array([nodes[i] for i in path[::-1]], int)

def arc_length_resample(points_um, ds_um):
    x=points_um[:,0]; y=points_um[:,1]
    seg=np.hypot(np.diff(x),np.diff(y)); s=np.concatenate([[0],np.cumsum(seg)])
    s_new=np.arange(0,s[-1],ds_um) if s[-1]>0 else np.array([0.0])
    x_new=np.interp(s_new,s,x); y_new=np.interp(s_new,s,y)
    return s_new,x_new,y_new

# ---------- 전극 모델 + 접촉 ----------
def evaluate_profile(img_path, material_id, valley_um, Rmin_um):
    sk = load_binary(img_path)
    pixels = longest_geodesic_path(sk)
    if pixels is None:
        return None
    xy = np.zeros((len(pixels),2), float)
    xy[:,0] = pixels[:,1]*px_um
    xy[:,1] = -pixels[:,0]*px_um
    s,x,y = arc_length_resample(xy, ds_um=px_um)
    ds_um = float(np.median(np.diff(s))) if len(s)>1 else px_um

    # 상부-포락선: valley bridging + min radius
    win = max(3, int(round(valley_um/ds_um)));  win += (win%2==0)
    y_close = minimum_filter1d(maximum_filter1d(y, size=win, mode="nearest"),
                               size=win, mode="nearest")
    sigma_um = (math.pi*Rmin_um)/(2*np.sqrt(2*np.log(2)))
    y_env = gaussian_filter1d(y_close, sigma=sigma_um/ds_um, mode="nearest")
    y_e = np.maximum(y_env, y)

    # gap-only contact with hysteresis + post
    gap = y_e - y
    w_loc = max(3, int(round(L_LOCAL_UM/ds_um)));  w_loc += (w_loc%2==0)
    gap_min = minimum_filter1d(gap, size=w_loc, mode="nearest")
    tol_on, tol_off = GAP_TOL_UM, GAP_TOL_UM * HYST_RATIO
    ok = np.zeros_like(gap_min, bool); state=False
    for i, g in enumerate(gap_min):
        if not state and g <= tol_on:
            state = True
        elif state and g >= tol_off:
            state = False
        ok[i] = state
    w_post = max(3, int(round(BRIDGE_POST_UM/ds_um)));  w_post += (w_post%2==0)
    ok = minimum_filter1d(maximum_filter1d(ok.astype(np.uint8), size=w_post, mode="nearest"),
                          size=w_post, mode="nearest").astype(bool)

    # 길이 및 x-span (strain은 나중에 baseline 기준으로 계산)
    x_span = float(abs(x[-1] - x[0]))                        # Δx
    elec_len = float(np.sum(np.hypot(np.diff(x), np.diff(y_e))))
    straight_len = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))   # 참고
    contact = float(ok.mean())

    return {
        "image": os.path.basename(img_path),
        "material_id": material_id,
        "valley_um": float(valley_um),
        "Rmin_um": float(Rmin_um),
        "x_span_um": x_span,
        "straight_length_um": straight_len,
        "electrode_length_um": elec_len,
        "contact_ratio": contact
    }

# ---------- 스윕 실행 ----------
rows = []
for img in IMAGES:
    if not os.path.exists(img):
        print("skip:", img)
        continue
    for mat in MATERIALS:
        mid = mat["id"]
        Rmin = mat["Rmin_um"]
        recs = []
        for valley in mat["valley_range"]:
            res = evaluate_profile(img, mid, valley, Rmin)
            if res:
                rows.append(res)
                recs.append(res)
        if recs:
            df_img_mat = pd.DataFrame(recs)
            df_img_mat.to_csv(
                os.path.join(OUT_DIR, "per_image",
                             f"{os.path.splitext(img)[0]}_{mid}.csv"),
                index=False
            )
            print("done:", img, mid, "rows:", len(recs))

# 합본 저장
df_all = pd.DataFrame(rows)
all_csv = os.path.join(OUT_DIR, "sweep_all_material.csv")
df_all.to_csv(all_csv, index=False)
print("저장:", all_csv, "rows:", len(df_all))

# ---------- baseline: Rigid, valley=1000에서 electrode_length 기준 ----------
base = (df_all[(df_all["material_id"]=="Rigid") & (df_all["valley_um"]==1000)]
        .groupby("image", as_index=True)["electrode_length_um"]
        .mean()
        .rename("L_elec_base_um"))

df_all = df_all.merge(base, left_on="image", right_index=True, how="left")
if df_all["L_elec_base_um"].isna().any():
    missing = df_all.loc[df_all["L_elec_base_um"].isna(),"image"].unique()
    raise ValueError(f"Baseline (Rigid, valley=1000) missing for images: {missing}")

df_all["strain_rigid_pct"] = (
    (df_all["electrode_length_um"] - df_all["L_elec_base_um"]) /
    df_all["L_elec_base_um"] * 100.0
)

# rebased 합본 저장
rebased_csv = os.path.join(OUT_DIR, "sweep_all_material_rebased.csv")
df_all.to_csv(rebased_csv, index=False)
print("rebased 저장:", rebased_csv)

# ---------- 평균 요약 (rigid 기준 strain) ----------
summary = (df_all
           .groupby(["material_id","valley_um","Rmin_um"], as_index=False)
           .agg(
               mean_strain_rigid_pct=("strain_rigid_pct","mean"),
               mean_contact_ratio=("contact_ratio","mean"),
               sd_contact_ratio=("contact_ratio","std"),
               n=("contact_ratio","count")
           ))

sum_csv = os.path.join(OUT_DIR, "sweep_summary_mean_material_rebased.csv")
summary.to_csv(sum_csv, index=False)
print("요약:", sum_csv)

# ---------- CI/Origin용 내보내기 ----------
summary["ci95"] = 1.96 * (summary["sd_contact_ratio"] /
                          np.sqrt(summary["n"].clip(lower=1)))
summary["lo"] = (summary["mean_contact_ratio"] - summary["ci95"]).clip(0,1)
summary["hi"] = (summary["mean_contact_ratio"] + summary["ci95"]).clip(0,1)

origin_csv = os.path.join(OUT_DIR, "sweep_summary_for_origin_material_rebased.csv")
summary.rename(columns={"mean_strain_rigid_pct": "mean_strain_pct_rigid"})[[
    "material_id","valley_um","Rmin_um",
    "mean_strain_pct_rigid","mean_contact_ratio",
    "sd_contact_ratio","n","ci95","lo","hi"
]].to_csv(origin_csv, index=False)
print("Origin용:", origin_csv)

# ---------- 그래프(평균 ±95% CI, rigid 기준 strain / 재료별) ----------
plt.figure(figsize=(8,4))
colors = {"PDMS":"tab:blue", "Parylene":"tab:orange", "PI":"tab:green", "Rigid":"tab:red"}

for mid in ["PDMS","Parylene","PI","Rigid"]:
    sub = summary[summary["material_id"]==mid].sort_values("mean_strain_rigid_pct")
    if len(sub)==0: continue
    x = sub["mean_strain_rigid_pct"].to_numpy()
    y = sub["mean_contact_ratio"].to_numpy()
    lo = sub["lo"].to_numpy()
    hi = sub["hi"].to_numpy()
    c = colors.get(mid, "gray")
    plt.plot(x, y, color=c, linewidth=2, label=f"{mid}")
    plt.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0)

plt.xlabel("Electrode strain vs Rigid (valley=1000) [%]")
plt.ylabel("Contact ratio (mean ± 95% CI)")
plt.title("Contact ratio vs strain (Rigid-baseline; PDMS / Parylene / PI / Rigid)")
plt.legend(frameon=False)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "cr_vs_strain_ci_material_rebased.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print("그림:", fig_path)
