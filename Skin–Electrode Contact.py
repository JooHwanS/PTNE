# === Skin–Electrode Contact (Gap-only, Final Jupyter) =========================
# 이진/그레이 이미지 → 스켈레톤 최장 경로 → µm 재표본화
# → 상부-포락선 전극(y_e): (좁은 골짜기 무시 + 최소 곡률반경)
# → contact map: gap(=y_e - y) 기반 판정(국소최소 + 히스테리시스 + 소거)
# 저장: out_follow_jupyter_gap/ 내 CSV/JSON/PNG

import os, math, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy.ndimage import (
    gaussian_filter1d, maximum_filter1d, minimum_filter1d, percentile_filter
)
# -------------------- 사용자 파라미터 --------------------
IMAGE_PATH       = "line1.tif"  # 입력 이미지
PIXELS_PER_UM    = 2.4          # px/µm
VALLEY_BRIDGE_UM = 10.0         # 이보다 좁은 골짜기는 건너뜀(클로징 폭)
R_MIN_UM         = 10.0         # 전극 최소 곡률 반경(µm) → 작을수록 유연

# 판정 보정(깜빡임/노이즈 완화)
GAP_TOL_UM       = 2.0          # 접촉 허용 갭(µm)
L_LOCAL_UM       = 3.0         # 국소 최소 gap 길이(µm)
HYST_RATIO       = 1.2       # 히스테리시스 off/on 비율(>=1.0)
BRIDGE_POST_UM   = 1.0          # OK 마스크 소거/메움 윈도(µm)

OUT_DIR = "out_follow_jupyter_gap"
# ---------------------------------------------------------

def load_binary(path):
    im = Image.open(path).convert("L")
    arr = np.array(im).astype(np.uint8)
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
    adj = [[] for _ in nodes]; deg = np.zeros(len(nodes),int)
    for i,(y,x) in enumerate(nodes):
        for ny,nx in neighbors8(y,x,H,W):
            if sk[ny,nx]:
                adj[i].append(idx[(ny,nx)])
        deg[i] = len(adj[i])
    return nodes, adj, deg

def bfs_longest_path_from(start_idx, adj):
    N=len(adj); dist=np.full(N,-1,int); prev=np.full(N,-1,int)
    q=deque([start_idx]); dist[start_idx]=0
    while q:
        u=q.popleft()
        for v in adj[u]:
            if dist[v]==-1: dist[v]=dist[u]+1; prev[v]=u; q.append(v)
    far = np.argmax(dist); return dist, prev, far

def longest_geodesic_path(sk):
    nodes, adj, deg = build_graph(sk)
    if not nodes: raise RuntimeError("Empty skeleton")
    endpoints = [i for i,d in enumerate(deg) if d==1] or [0]
    _,_,farA = bfs_longest_path_from(endpoints[0], adj)
    _,prev2,farB = bfs_longest_path_from(farA, adj)
    path=[]; cur=farB
    while cur!=-1:
        path.append(cur)
        if cur==farA: break
        cur=prev2[cur]
    return np.array([nodes[i] for i in path[::-1]], int)  # (row,col)

def arc_length_resample(points_um, ds_um):
    x=points_um[:,0]; y=points_um[:,1]
    seg=np.hypot(np.diff(x), np.diff(y)); s=np.concatenate([[0],np.cumsum(seg)])
    s_new=np.arange(0, s[-1], ds_um) if s[-1]>0 else np.array([0.0])
    x_new=np.interp(s_new, s, x); y_new=np.interp(s_new, s, y)
    return s_new, x_new, y_new

# 1) 입력→스켈레톤→경로
os.makedirs(OUT_DIR, exist_ok=True)
px_um = 1.0/PIXELS_PER_UM
sk = load_binary(IMAGE_PATH)
pixels = longest_geodesic_path(sk)

# 2) 좌표 변환(y-up) + 재표본화(µm)
xy_um = np.zeros((len(pixels),2), float)
xy_um[:,0] = pixels[:,1]*px_um
xy_um[:,1] = -pixels[:,0]*px_um
s, x, y = arc_length_resample(xy_um, ds_um=px_um)
ds_um = np.median(np.diff(s)) if len(s)>1 else px_um

# 3) 전극 상부-포락선: 좁은 골짜기 무시 + 최소 곡률반경
#    3.1 Valley bridging (1D grayscale closing)
win = max(3, int(round(VALLEY_BRIDGE_UM/ds_um)));  win += (win%2==0)
y_close = minimum_filter1d(maximum_filter1d(y, size=win, mode="nearest"),
                           size=win, mode="nearest")
#    3.2 Min radius → 충분히 부드러운 LP (λ≈πR 보수적 매핑)
sigma_um = (math.pi*R_MIN_UM)/(2*np.sqrt(2*np.log(2)))
y_env = gaussian_filter1d(y_close, sigma=sigma_um/ds_um, mode='nearest')
#    3.3 피부 내부 침투 금지
y_e = np.maximum(y_env, y)

# 4) gap-only contact decision (robust)
gap = y_e - y  # y-up 좌표이므로 접촉이면 0에 가까움

#    4.1 국소 최소로 미세 진동 억제
w_loc = max(3, int(round(L_LOCAL_UM/ds_um)));  w_loc += (w_loc%2==0)
gap_min = minimum_filter1d(gap, size=w_loc, mode="nearest")

#    4.2 히스테리시스(깜빡임 방지)
tol_on  = GAP_TOL_UM
tol_off = GAP_TOL_UM * HYST_RATIO
ok = np.zeros_like(gap_min, dtype=bool)
state = False
for i, g in enumerate(gap_min):
    if not state and g <= tol_on:
        state = True
    elif state and g >= tol_off:
        state = False
    ok[i] = state

#    4.3 마스크 소거/메움(점상 구멍 제거)
w_post = max(3, int(round(BRIDGE_POST_UM/ds_um)));  w_post += (w_post%2==0)
ok = minimum_filter1d(maximum_filter1d(ok.astype(np.uint8), size=w_post, mode="nearest"),
                      size=w_post, mode="nearest").astype(bool)

contact_ratio = float(ok.mean())

# 5) 저장
df = pd.DataFrame({
    "s_um": s, "x_um": x, "y_um": y,
    "y_electrode_um": y_e,
    "gap_um": gap, "gap_min_loc_um": gap_min,
    "contact_ok": ok.astype(int)
})
df.to_csv(os.path.join(OUT_DIR, "profile_follow_contact.csv"), index=False)

meta = {
    "pixels_per_um": PIXELS_PER_UM, "pixel_size_um": px_um,
    "VALLEY_BRIDGE_UM": VALLEY_BRIDGE_UM, "R_MIN_UM": R_MIN_UM,
    "GAP_TOL_UM": GAP_TOL_UM, "L_LOCAL_UM": L_LOCAL_UM,
    "HYST_RATIO": HYST_RATIO, "BRIDGE_POST_UM": BRIDGE_POST_UM,
    "contact_ratio": contact_ratio, "input_image": os.path.basename(IMAGE_PATH)
}
with open(os.path.join(OUT_DIR, "follow_summary.json"), "w") as f:
    json.dump(meta, f, indent=2)

# 6) 플롯
plt.figure(figsize=(10,3))
plt.plot(x, y, label="skin", linewidth=2)
plt.plot(x, y_e, label="electrode", linewidth=3)
plt.title("Moderately flexible electrode vs skin")
plt.xlabel("x (um)"); plt.ylabel("y (um)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f01_profile_vs_electrode.png"))
plt.show()

plt.figure(figsize=(10,3))
ok_idx=np.where(ok)[0]; ng_idx=np.where(~ok)[0]
plt.plot(x[ok_idx], y[ok_idx], '.', label="OK")
if len(ng_idx): plt.plot(x[ng_idx], y[ng_idx], '.', label="NG")
plt.title(f"Contact map  OK={contact_ratio:.1%} (gap≤{GAP_TOL_UM}µm, hysteresis×{HYST_RATIO})")
plt.xlabel("x (um)"); plt.ylabel("y (um)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f02_contact_map.png"))
plt.show()

print("완료:", OUT_DIR, "| 예측 접촉률:", f"{contact_ratio:.2%}")
# --- Export plot data for Origin/Excel ---
plot_df = pd.DataFrame({
    "x_um": x,
    "y_skin_um": y,
    "y_electrode_um": y_e,
    "gap_um": gap,
    "contact_ok": ok.astype(int)
})
plot_csv = os.path.join(OUT_DIR, "skin_electrode_profile.csv")
plot_df.to_csv(plot_csv, index=False)

# Electrode polyline only
elec_df = pd.DataFrame({"x_um": x, "y_electrode_um": y_e})
elec_csv = os.path.join(OUT_DIR, "electrode_polyline.csv")
elec_df.to_csv(elec_csv, index=False)

print("데이터 저장:", plot_csv, "|", elec_csv)

# --- Export electrode-only curve as image (transparent) and SVG ---
# PNG(투명) — 논문 본문/포스터용
plt.figure(figsize=(10,1.8))
plt.plot(x, y_e, linewidth=3, color="tab:orange")
plt.axis("off")
plt.gca().set_aspect("auto")
png_path = os.path.join(OUT_DIR, "electrode_curve_only.png")
plt.savefig(png_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close()

# SVG(벡터) — Origin/Illustrator에서 선으로 편집 가능
plt.figure(figsize=(10,1.8))
plt.plot(x, y_e, linewidth=1.5, color="black")
plt.axis("off")
svg_path = os.path.join(OUT_DIR, "electrode_curve_only.svg")
plt.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close()

print("전극선 이미지:", png_path, "|", svg_path)
