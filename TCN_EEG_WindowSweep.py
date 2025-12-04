# 필요시: !pip install scikit-learn tensorflow

import numpy as np, pandas as pd, re, os, warnings
warnings.filterwarnings("ignore", category=UserWarning)

from IPython.display import display
import ipywidgets as widgets
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter, defaultdict
from secrets import randbits

from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ===================== [사용자 설정] =====================
FS = 200
EXCLUDE_EDGE_SEC = 5           # 각 파일 앞/뒤 컷(초)
F_MIN, F_MAX = 1, 20.0       # PSD 범위(Hz)

# QC — 진폭 피크만
USE_QC_FILTER  = True
AMP_PEAK_MAX   = 400.0

# 학습용 세션 z-score (저장물에는 미적용)
APPLY_ZSCORE_FOR_TRAIN = True

# 시퀀스 스택(PSD 프레임 T개)
SEQ_T   = 5                    # 3 또는 5 권장
SEQ_HOP = 1

# TCN window-length sweep 고정 파라미터
SWEEP_WIN_LIST = [1, 2, 4, 8, 16, 20, 24, 32]  # 초
SWEEP_OVERLAP  = 0.50
SWEEP_SEG_SEC  = 2             # Welch 세그먼트 길이(초) → Δf=0.5 Hz
SWEEP_RUNS     = 10
RANDOM_SEED    = None

# 라벨 파싱 실패 시 수동 매핑(옵션)
FALLBACK_LABEL_MAP = {}

# 업로드 위젯 개수(폴더 수)
FOLDERS = 5

# ================== 유틸 ====================
def _derived(win_sec, overlap, nperseg_sec, nover_ratio=0.5):
    hop_sec    = max(1e-9, win_sec*(1.0 - overlap))
    W_NPERSEG  = int(max(8, nperseg_sec * FS))
    W_NOVERLAP = int(W_NPERSEG * nover_ratio)
    DELTA_F    = FS / W_NPERSEG
    return hop_sec, W_NPERSEG, W_NOVERLAP, DELTA_F

def parse_stream_txt_bytes(b):
    if isinstance(b, memoryview): b = b.tobytes()
    text = b.decode("utf-8", errors="ignore")
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        for v in s.replace(",", " ").split():
            try: vals.append(float(v))
            except: pass
    return np.array(vals, float)

def stem(path): return os.path.basename(path).rsplit(".", 1)[0]

def trim_edges(sig):
    off = int(EXCLUDE_EDGE_SEC * FS)
    return sig[off:-off] if len(sig) > 2 * off else np.array([], float)

def split_windows(sig, win_sec, overlap):
    hop_sec = max(1e-9, win_sec * (1.0 - overlap))
    win_len = int(win_sec * FS)
    hop = max(1, int(hop_sec * FS))
    if len(sig) < win_len: return []
    return [sig[i:i + win_len] for i in range(0, len(sig) - win_len + 1, hop)]

def infer_label(name):
    s = name.lower()
    m = re.search(r'(\d)\s*[-_ ]*\s*back', s) or re.search(r'back\s*[-_ ]*(\d)', s)
    if m:
        g = [g for g in m.groups() if g is not None][0]
        return int(g)
    base = stem(name)
    return FALLBACK_LABEL_MAP.get(name, FALLBACK_LABEL_MAP.get(base, None))

def qc_basic(x):  # 원시 윈도우 기준
    return np.all(np.isfinite(x)) and not np.any(np.abs(x) > AMP_PEAK_MAX)

def safe_split_indices(y_labels, test_ratio=0.2, seed=None):
    n = len(y_labels)
    n_test = max(1, int(round(n * test_ratio)))
    if n - n_test < 1: n_test = n - 1
    cnt = Counter(y_labels)
    can_strat = (min(cnt.values()) >= 2) and (n_test >= len(cnt))
    idx_all = np.arange(n)
    tr_idx, te_idx = train_test_split(idx_all, test_size=n_test, random_state=seed,
                                      stratify=(y_labels if can_strat else None))
    return tr_idx, te_idx

def compute_psd_log_from_windows(windows, fs, w_nperseg, w_noverlap, f_ref=None, band_mask=None):
    psd_rows = []
    for w in windows:
        x = w - np.mean(w)
        nps = min(w_nperseg, len(x))
        nov = min(w_noverlap, max(0, nps // 2))
        f, Pxx = welch(x, fs=fs, nperseg=nps, noverlap=nov, scaling="density")
        if f_ref is None:
            f_ref = f.copy()
            band_mask = (f_ref >= F_MIN) & (f_ref <= F_MAX)
        if len(f) != len(f_ref) or not np.allclose(f, f_ref):
            Pxx = np.interp(f_ref, f, Pxx)
        psd_rows.append(Pxx[band_mask])
    X_log = np.log10(np.vstack(psd_rows) + 1e-12)
    return X_log, f_ref[band_mask], f_ref, band_mask

def build_sequences(names, X_log_train, y_labels, T=5, hop=1):
    sess, order = [], []
    for nm in names:
        if "_w" in nm:
            s, w = nm.rsplit("_w", 1)
            sess.append(s)
            try: order.append(int(w))
            except: order.append(0)
        else:
            sess.append(nm); order.append(0)
    df = pd.DataFrame({"sess": sess, "ord": order, "idx": np.arange(len(names))})
    X_list, y_list = [], []
    half = T // 2
    for s, g in df.sort_values(["sess", "ord"]).groupby("sess", sort=False):
        idxs = g["idx"].to_numpy()
        if len(idxs) < T: continue
        for j in range(half, len(idxs) - half, hop):
            neigh = idxs[j - half: j + half + 1]
            if len(neigh) != T: continue
            X_list.append(neigh)
            y_list.append(y_labels[idxs[j]])
    if not X_list: return None, None
    return np.stack(X_list, 0), np.array(y_list, int)

# ===== TCN 모델 =====
def build_tcn(n_time, n_feat, n_cls):
    inp = keras.Input(shape=(n_time, n_feat))
    x = inp
    for d in [1, 2, 4, 8]:
        y = layers.Conv1D(64, 3, padding="causal", dilation_rate=d, use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Dropout(0.2)(y)
        y = layers.Conv1D(64, 3, padding="causal", dilation_rate=d, use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        if x.shape[-1] != 64:
            x = layers.Conv1D(64, 1, padding="same")(x)
        x = layers.Add()([x, y])
        x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# ===================== 위젯 =======================
uploads = [widgets.FileUpload(accept=".txt", multiple=True, description=f"폴더{i+1} 업로드") for i in range(FOLDERS)]
for u in uploads: display(u)
btn = widgets.Button(description="TCN window-length sweep (10 runs)", button_style="primary")
msg = widgets.HTML(); display(btn, msg)
out = widgets.Output(); display(out)

# ===================== 실행 =======================
@out.capture(clear_output=True)
def run(_):
    msg.value = "실행 중…"
    try:
        # 업로드 수집
        file_groups = {}
        for u in uploads:
            items = list(u.value.values()) if isinstance(u.value, dict) else list(u.value)
            for v in items:
                if isinstance(v, tuple): v = v[1]
                if isinstance(v, dict) and "content" in v and "name" in v:
                    name = v["name"]
                    file_groups.setdefault(name, []).append(v["content"])
        if not file_groups:
            msg.value = "<span style='color:red'>업로드된 파일이 없습니다.</span>"; return

        # 같은 이름 파일 병합(raw, 앞뒤 5초 컷)
        merged_raw = {}
        for name, parts in file_groups.items():
            segs = []
            for p in parts:
                sig = parse_stream_txt_bytes(p); sig = trim_edges(sig)
                if len(sig) > 0: segs.append(sig)
            if segs: merged_raw[name] = np.concatenate(segs)
        if not merged_raw:
            msg.value = "<span style='color:red'>병합된 신호 없음.</span>"; return

        # ===== TCN window-length sweep =====
        sweep_rows = []
        for wsec in SWEEP_WIN_LIST:
            _, WNPs, WNOs, DELTAs = _derived(wsec, SWEEP_OVERLAP, SWEEP_SEG_SEC, 0.5)

            # 1) 윈도우+QC
            wins, labs, names = [], [], []
            counters = defaultdict(int)
            for name, sig in merged_raw.items():
                lab = infer_label(name)
                if lab is None or not (0 <= lab <= 4): continue
                sname = stem(name)
                for w in split_windows(sig, wsec, SWEEP_OVERLAP):
                    if (not USE_QC_FILTER) or qc_basic(w):
                        counters[sname] += 1
                        wins.append(w); labs.append(lab); names.append(f"{sname}_w{counters[sname]:03d}")

            if len(wins) == 0:
                sweep_rows.append(dict(window_sec=wsec, n_sequences=0,
                                       acc_mean=np.nan, acc_std=np.nan, f1_mean=np.nan, f1_std=np.nan))
                print(f"[SWEEP] win={wsec}s → n=0"); continue

            # 2) PSD(log) (z-score 토글)
            if APPLY_ZSCORE_FOR_TRAIN:
                sess_stats = {stem(n): (float(np.mean(sig)), float(np.std(sig)+1e-8)) for n, sig in merged_raw.items()}
                z_wins = []
                for nm, w in zip(names, wins):
                    sname = nm.split("_w")[0]
                    m, s = sess_stats.get(sname, (0.0, 1.0))
                    z_wins.append((w - m) / s)
                X_log, _, _, _ = compute_psd_log_from_windows(z_wins, FS, WNPs, WNOs)
            else:
                X_log, _, _, _ = compute_psd_log_from_windows(wins, FS, WNPs, WNOs)
            y = np.array(labs, int)

            # 3) 시퀀스 스택
            seq_idx_s, y_seq_s = build_sequences(names, X_log, y, T=SEQ_T, hop=SEQ_HOP)
            if seq_idx_s is None:
                sweep_rows.append(dict(window_sec=wsec, n_sequences=0,
                                       acc_mean=np.nan, acc_std=np.nan, f1_mean=np.nan, f1_std=np.nan))
                print(f"[SWEEP] win={wsec}s → seq=0"); continue

            # 표준화 후 [n_seq, T, n_freq]
            scaler_s = StandardScaler()
            X_flat_s2 = scaler_s.fit_transform(X_log)
            X_seq_s = np.stack([X_flat_s2[idxs, :] for idxs in seq_idx_s], axis=0)

            # 클래스 가중치
            classes_s = np.unique(y_seq_s)
            cw_vals_s = compute_class_weight(class_weight="balanced", classes=classes_s, y=y_seq_s)
            cw_dict_s = {int(k): float(v) for k, v in zip(classes_s, cw_vals_s)}

            # 4) 10회 반복 평가 (TCN 고정)
            n_classes_s = len(classes_s)
            n_time_s, n_feat_s = X_seq_s.shape[1], X_seq_s.shape[2]
            acc_list, f1_list = [], []
            for r in range(SWEEP_RUNS):
                seed = (RANDOM_SEED if RANDOM_SEED is not None else randbits(32)) + r
                tr, te = safe_split_indices(y_seq_s, 0.2, seed)
                Xtr, Xte = X_seq_s[tr], X_seq_s[te]
                ytr, yte = y_seq_s[tr], y_seq_s[te]

                model = build_tcn(n_time_s, n_feat_s, n_classes_s)
                es = callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
                rlrop = callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, verbose=0)
                model.fit(Xtr, ytr, epochs=60, batch_size=64, verbose=0,
                          validation_split=0.2, class_weight=cw_dict_s,
                          callbacks=[es, rlrop])

                yhat = model.predict(Xte, verbose=0).argmax(axis=1)
                acc_list.append(accuracy_score(yte, yhat))
                f1_list.append(f1_score(yte, yhat, average="macro"))

            sweep_rows.append(dict(
                window_sec=wsec,
                n_sequences=len(y_seq_s),
                acc_mean=float(np.mean(acc_list)), acc_std=float(np.std(acc_list)),
                f1_mean=float(np.mean(f1_list)),  f1_std=float(np.std(f1_list))
            ))
            print(f"[SWEEP] win={wsec:>2}s → acc={np.mean(acc_list):.3f}±{np.std(acc_list):.3f}, "
                  f"f1={np.mean(f1_list):.3f}±{np.std(f1_list):.3f}, seq={len(y_seq_s)}")

        sweep_df = pd.DataFrame(sweep_rows).sort_values("window_sec")
        sweep_out = "tcn_win_sweep_10runs.csv"
        sweep_df.to_csv(sweep_out, index=False)
        print(f"[Saved] {sweep_out}")

        msg.value = (
            f"<b>완료</b> — TCN window sweep만 수행"
            f"<br>저장: {sweep_out} (columns: window_sec, n_sequences, acc_mean/acc_std, f1_mean/f1_std)"
        )

    except Exception as e:
        msg.value = f"<span style='color:red'>Error: {type(e).__name__}: {e}</span>"

btn.on_click(run)
