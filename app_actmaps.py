import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
import pandas as pd
import gdown
import os

# ======================= CONFIG =======================

DNN_DATASET_ROOT = Path("DFG_data/dnn_datasets")
RIDGE_MEAN_WEIGHTS_PATH = DNN_DATASET_ROOT / "ridge_actmaps_stimXcog.pkl"
RIDGE_VAR_WEIGHTS_PATH = DNN_DATASET_ROOT / "ridge_var_actmaps_stimXcog.pkl"

SCORES_CSV = Path("DFG_data/DFG_behavioural_results.csv")
SUBJECT_COL = "Subject"

CHUNK_SIZE = 256
RESET_SESSIONS = [1, 2, 3]

# Google Drive file IDs - Extracted from your Google Drive links
# Format: Just the file ID, not the full URL
GDRIVE_FILES = {
    "ridge_actmaps_stimXcog.pkl": "1Tf8juB2VDKtj3CRQ4vfIh_ZytO_DrlZf",
    "ridge_var_actmaps_stimXcog.pkl": "1_Q4Yw_OavZi9byvAOUqC-R899FbOLWFd",
    "DFG_behavioural_results.csv": "1gACthxLRGxbwOmGwp3WMz529NrcYFdJ6",
    # Session pickle files
    "session_1_cog2bold.pkl": "1bZuJ5NylCysczdvNWxPR7pGmsIBSoivS",
    "session_2_cog2bold.pkl": "1gsq9RF-zu4c9YZ1WtDo8IaUa-0UxxxjD",
    "session_3_cog2bold.pkl": "1EPZF0HDmcEK2SnlDKRty2DxZCCV6gzBA",
}


# ======================= GOOGLE DRIVE DOWNLOAD =======================

def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, str(output_path), quiet=False, fuzzy=True)


def ensure_data_files():
    """
    Check if required data files exist. If not, download from Google Drive.
    Call this before loading any data.
    """
    files_to_check = {}
    
    # Check required model files
    for filename in ["ridge_actmaps_stimXcog.pkl", "ridge_var_actmaps_stimXcog.pkl", "DFG_behavioural_results.csv"]:
        file_path = str(RIDGE_MEAN_WEIGHTS_PATH.parent.parent / filename) if filename.endswith('.csv') else str(DNN_DATASET_ROOT / filename)
        if filename == "DFG_behavioural_results.csv":
            file_path = str(SCORES_CSV)
        
        if not Path(file_path).exists():
            file_id = GDRIVE_FILES.get(filename)
            if file_id and not file_id.startswith("YOUR_FILE_ID"):
                files_to_check[file_path] = file_id
            else:
                st.error(f"❌ Missing file: `{filename}`")
                st.error("Please update the `GDRIVE_FILES` dictionary in the code with your Google Drive file IDs.")
                st.info("📖 See README.md for instructions on how to get Google Drive file IDs.")
                st.stop()
    
    # Check for at least one session pickle file
    session_pickles = sorted(DNN_DATASET_ROOT.glob("session_*_cog2bold.pkl"))
    if not session_pickles:
        # Try to download session files
        session_files = [k for k in GDRIVE_FILES.keys() if k.startswith("session_") and k.endswith("_cog2bold.pkl")]
        if session_files:
            for sess_file in session_files:
                file_path = str(DNN_DATASET_ROOT / sess_file)
                file_id = GDRIVE_FILES[sess_file]
                if not file_id.startswith("YOUR_FILE_ID"):
                    files_to_check[file_path] = file_id
        else:
            st.error("❌ No session pickle files found and none configured in GDRIVE_FILES")
            st.error("Please add at least one `session_*_cog2bold.pkl` file to GDRIVE_FILES")
            st.stop()
    
    # Download missing files
    if files_to_check:
        st.info(f"📥 Downloading {len(files_to_check)} file(s) from Google Drive...")
        progress_bar = st.progress(0)
        
        for idx, (file_path, file_id) in enumerate(files_to_check.items()):
            filename = Path(file_path).name
            st.write(f"⏳ Downloading `{filename}`...")
            try:
                download_from_gdrive(file_id, file_path)
                st.success(f"✅ Downloaded `{filename}`")
            except Exception as e:
                st.error(f"❌ Failed to download `{filename}`: {e}")
                st.error("Make sure the file is publicly accessible or shared with 'Anyone with the link'")
                st.stop()
            progress_bar.progress((idx + 1) / len(files_to_check))
        
        st.success("✅ All files downloaded successfully!")
        st.balloons()
        st.info("🔄 Reloading app...")
        st.rerun()


# ======================= CACHED LOADERS =======================

@st.cache_resource
def load_masker_and_metadata():
    """Load masker from session pickles. Cached to run only once."""
    pkl_candidates = sorted(DNN_DATASET_ROOT.glob("session_*_cog2bold.pkl"))
    if not pkl_candidates:
        st.error(f"No session_*_cog2bold.pkl found in {DNN_DATASET_ROOT}")
        st.stop()

    pkl_path = pkl_candidates[0]
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    meta = bundle["meta"]
    masker = meta["masker"]
    
    return masker


@st.cache_resource
def load_models():
    """Load the Ridge models for activation maps. Cached to run only once."""
    if not RIDGE_MEAN_WEIGHTS_PATH.exists():
        st.error(f"Mean model file not found: {RIDGE_MEAN_WEIGHTS_PATH}")
        st.stop()
    if not RIDGE_VAR_WEIGHTS_PATH.exists():
        st.error(f"Variance model file not found: {RIDGE_VAR_WEIGHTS_PATH}")
        st.stop()

    with open(RIDGE_MEAN_WEIGHTS_PATH, "rb") as f:
        mean_dict = pickle.load(f)
    with open(RIDGE_VAR_WEIGHTS_PATH, "rb") as f:
        var_dict = pickle.load(f)

    return mean_dict, var_dict


@st.cache_data
def load_behavioral_data():
    """Load behavioral CSV and extract relevant statistics."""
    df = pd.read_csv(SCORES_CSV)
    if SUBJECT_COL not in df.columns:
        st.error(f"Subject column '{SUBJECT_COL}' not found in {SCORES_CSV}")
        st.stop()
    return df


@st.cache_data
def get_reset_statistics(_df, cog_base_cols, _target_sessions):
    """
    Compute global and per-session means from behavioral CSV.
    Uses the actual cognitive feature columns from the trained model.
    """
    n_cog = len(cog_base_cols)
    
    # Global mean: average across all subjects
    global_stats = np.zeros(n_cog, dtype=np.float64)
    for i, base_col in enumerate(cog_base_cols):
        # Try session-specific columns first, then fall back to base
        col_vals = []
        for sess in _target_sessions:
            sess_col = f"{base_col}#{sess}"
            if sess_col in _df.columns:
                col_vals.extend(_df[sess_col].dropna().tolist())
        if not col_vals and base_col in _df.columns:
            col_vals = _df[base_col].dropna().tolist()
        
        if col_vals:
            global_stats[i] = np.mean(col_vals)
    
    # Per-session means
    session_stats = {}
    for sess in _target_sessions:
        sess_stats = np.zeros(n_cog, dtype=np.float64)
        for i, base_col in enumerate(cog_base_cols):
            sess_col = f"{base_col}#{sess}"
            if sess_col in _df.columns:
                vals = _df[sess_col].dropna()
                if len(vals) > 0:
                    sess_stats[i] = vals.mean()
                else:
                    sess_stats[i] = global_stats[i]
            elif base_col in _df.columns:
                vals = _df[base_col].dropna()
                if len(vals) > 0:
                    sess_stats[i] = vals.mean()
                else:
                    sess_stats[i] = global_stats[i]
            else:
                sess_stats[i] = global_stats[i]
        session_stats[sess] = sess_stats
    
    return global_stats, session_stats


# ======================= HELPERS =======================

def get_default_hrf_weights(n_trs):
    """Default HRF-like weights for temporal aggregation."""
    base = np.array([0.0, 0.15, 1.0, 0.8, 0.65, 0.35, 0.15, 0.05], dtype=np.float64)
    if n_trs <= len(base):
        w = base[:n_trs].copy()
    else:
        w = np.concatenate([base, np.full(n_trs - len(base), base[-1])])
    if w.sum() > 0: 
        w /= w.sum()
    return w


def build_design_matrix(
    cog_raw: np.ndarray,           # (N, F)
    global_stim_ids: np.ndarray,   # (N,)
    cog_mean: np.ndarray,          # (F,)
    cog_std: np.ndarray,           # (F,)
    n_stimuli: int,
) -> np.ndarray:
    """
    Build design matrix Z: [K blocks of F standardized cog, K one-hot, bias].
    D_in = F*K + K + 1
    """
    cog = (cog_raw - cog_mean) / cog_std
    N, F = cog.shape
    stim_onehot = np.eye(n_stimuli, dtype=np.float64)[global_stim_ids.astype(np.int64, copy=False)]
    cog_blocks = (cog[:, None, :] * stim_onehot[:, :, None]).reshape(N, n_stimuli * F)
    bias = np.ones((N, 1), dtype=np.float64)
    Z = np.concatenate([cog_blocks, stim_onehot, bias], axis=1)
    return Z


def build_z(cog_raw, stim_label, cog_mean, cog_std, label2id, n_cog, n_stim):
    """Build design matrix for a single sample."""
    s_idx = label2id[stim_label]
    Z = build_design_matrix(
        cog_raw=cog_raw[np.newaxis, :],
        global_stim_ids=np.array([s_idx], dtype=np.int64),
        cog_mean=cog_mean,
        cog_std=cog_std,
        n_stimuli=n_stim,
    )
    return Z[0]


def aggregate_over_tr(values_2d, mode, tr_index, hrf_weights, custom_weights_str):
    """Aggregate activation values over TRs."""
    V, T = values_2d.shape
    if mode == "no_agg":
        idx = int(np.clip(tr_index, 0, T - 1))
        return values_2d[:, idx], f"TR={idx}"
    elif mode == "hrf":
        return values_2d @ hrf_weights, "HRF-weighted"
    elif mode == "custom":
        try:
            parts = [float(p) for p in custom_weights_str.split(",") if p.strip()]
            w = np.array(parts, dtype=np.float64)
            if len(w) < T:
                w = np.concatenate([w, np.zeros(T - len(w))])
            else:
                w = w[:T]
            if w.sum() > 0: 
                w /= w.sum()
            return values_2d @ w, "Custom-weighted"
        except:
            return np.zeros(V), "Error parsing weights"
    elif mode == "max":
        idx = np.abs(values_2d).argmax(axis=1)
        return values_2d[np.arange(V), idx], "Max over TRs"
    return values_2d[:, 0], "Unknown"


# ======================= MAIN UI =======================

def main():
    st.set_page_config(layout="wide", page_title="fMRI Activation Maps Explorer")

    # 0. ENSURE DATA FILES (download from Google Drive if needed)
    ensure_data_files()

    # 1. LOAD DATA
    with st.spinner("Loading models..."):
        masker = load_masker_and_metadata()
        mean_dict, var_dict = load_models()
        df_behavioral = load_behavioral_data()

        # Unpack model data
        W = mean_dict["W"]                          # (D_in, V, T)
        W_var = var_dict["W_var"]                   # (D_in, V, T)
        cog_base_cols = mean_dict["cog_base_cols"]
        global_labels = mean_dict["stimulus_labels"]
        n_stim = mean_dict["n_stimuli"]
        cog_mean = mean_dict["cog_mean"]
        cog_std = mean_dict["cog_std"]
        n_cog = len(cog_base_cols)
        n_trs = W.shape[2]
        
        label2id = {lab: i for i, lab in enumerate(global_labels)}

        # Load Reset Stats from behavioral CSV using model's cognitive features
        global_stats, sess_stats = get_reset_statistics(
            df_behavioral, cog_base_cols, RESET_SESSIONS
        )

    # 2. INIT STATE
    if "cog_sliders" not in st.session_state:
        st.session_state.cog_sliders = global_stats.copy()
    if "slider_ranges" not in st.session_state:
        # Compute ranges once and store them
        ranges = []
        for i in range(n_cog):
            mu = global_stats[i]
            sigma = cog_std[i]
            if sigma <= 0:
                low, high = mu - 1.0, mu + 1.0
            else:
                low, high = mu - 3 * sigma, mu + 3 * sigma
            low = max(0, low)
            ranges.append((float(low), float(high)))
        st.session_state.slider_ranges = ranges
    if "vmin" not in st.session_state: 
        st.session_state.vmin = -2.0
    if "vmax" not in st.session_state: 
        st.session_state.vmax = 2.0

    st.title("fMRI Activation Maps Viewer")

    # =========================================================
    # SPLIT SCREEN: LEFT (Settings) | RIGHT (Sliders)
    # =========================================================
    col_left, col_right = st.columns([1, 2])

    # ------------------ LEFT COLUMN (SECTION 1) ------------------
    with col_left:
        with st.expander("🛠️ 1. Settings", expanded=True):
            st.markdown("##### Analysis Mode")
            mode = st.radio(
                "Mode", 
                ["Single stimulus", "Variance map", "Contrast (t-like)"], 
                label_visibility="collapsed"
            )

            if mode == "Single stimulus":
                stim = st.selectbox("Stimulus", global_labels)
            elif mode == "Variance map":
                stim = st.selectbox("Stimulus", global_labels)
            else:
                stimA = st.selectbox("Stim A", global_labels, index=0)
                stimB = st.selectbox(
                    "Stim B", 
                    global_labels, 
                    index=1 if len(global_labels) > 1 else 0
                )

            st.markdown("---")
            st.markdown("##### Time Aggregation")
            agg_mode = st.selectbox("Method", ["hrf", "no_agg", "custom", "max"])

            custom_w_str = ""
            if agg_mode == "custom":
                custom_w_str = st.text_input("Weights", "0,0.2,1,0.6,0.3")

    # ------------------ RIGHT COLUMN (SECTION 2) ------------------
    with col_right:
        with st.expander("🎛️ 2. Cognitive Feature Sliders", expanded=True):

            # --- RESET TOOLBAR ---
            c_reset1, c_reset2 = st.columns([2, 1])
            with c_reset1:
                reset_choice = st.selectbox(
                    "Reset Source",
                    ["Global Mean", "Session 1", "Session 2", "Session 3"],
                    label_visibility="collapsed"
                )
            with c_reset2:
                if st.button("Reset Now", use_container_width=True):
                    if reset_choice == "Global Mean":
                        st.session_state.cog_sliders = global_stats.copy()
                    elif reset_choice == "Session 1":
                        st.session_state.cog_sliders = sess_stats[1].copy()
                    elif reset_choice == "Session 2":
                        st.session_state.cog_sliders = sess_stats[2].copy()
                    elif reset_choice == "Session 3":
                        st.session_state.cog_sliders = sess_stats[3].copy()
                    st.rerun()

            st.markdown("---")

            # --- SLIDERS ---
            grid_cols = st.columns(2)

            for i, name in enumerate(cog_base_cols):
                low, high = st.session_state.slider_ranges[i]
                col_idx = i % 2
                with grid_cols[col_idx]:
                    val = st.slider(
                        f"{name}",
                        low, high,
                        float(st.session_state.cog_sliders[i]),
                        step=(high - low) / 100.0,
                        key=f"slider_{i}"
                    )
                    st.session_state.cog_sliders[i] = val

    # =========================================================
    # SECTION 3: BRAIN MAP (BOTTOM - FULL WIDTH)
    # =========================================================
    st.markdown("---")
    st.subheader("🧠 3. Brain Activation Map")

    # --- DYNAMIC CONTROLS (TR SLIDER) ---
    tr_val = 0
    if agg_mode == "no_agg":
        st.info("Time Series Mode: Select TR to view")
        tr_val = st.slider("TR Index", 0, n_trs - 1, 0)

    # --- CALCULATION ---
    cog_raw_current = st.session_state.cog_sliders
    hrf_w = get_default_hrf_weights(n_trs)
    display_map = None
    title_str = ""

    if mode == "Single stimulus":
        z = build_z(cog_raw_current, stim, cog_mean, cog_std, label2id, n_cog, n_stim)
        Y_hat = np.tensordot(z, W, axes=(0, 0))  # (V, T)
        display_map, agg_info = aggregate_over_tr(
            Y_hat, agg_mode, tr_val, hrf_w, custom_w_str
        )
        title_str = f"Activation Map: {stim} ({agg_info})"
    elif mode == "Variance map":
        z = build_z(cog_raw_current, stim, cog_mean, cog_std, label2id, n_cog, n_stim)
        Y_var = np.tensordot(z, W_var, axes=(0, 0))  # (V, T)
        display_map, agg_info = aggregate_over_tr(
            Y_var, agg_mode, tr_val, hrf_w, custom_w_str
        )
        title_str = f"Variance Map: {stim} ({agg_info})"
    else:
        zA = build_z(cog_raw_current, stimA, cog_mean, cog_std, label2id, n_cog, n_stim)
        zB = build_z(cog_raw_current, stimB, cog_mean, cog_std, label2id, n_cog, n_stim)
        muA = np.tensordot(zA, W, axes=(0, 0))       # (V, T)
        muB = np.tensordot(zB, W, axes=(0, 0))       # (V, T)
        varA = np.tensordot(zA, W_var, axes=(0, 0))  # (V, T)
        varB = np.tensordot(zB, W_var, axes=(0, 0))  # (V, T)
        se = np.sqrt(np.maximum(varA, 1e-8) + np.maximum(varB, 1e-8))
        display_map, agg_info = aggregate_over_tr(
            (muA - muB) / se, agg_mode, tr_val, hrf_w, custom_w_str
        )
        title_str = f"Contrast: {stimA} vs {stimB} ({agg_info})"

    # --- VISUALIZATION CONTROLS ---
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        nslices = st.number_input("# Slices (0=Auto)", 0, 30, 7)
    with c2:
        vmin_input = st.number_input("vmin", value=st.session_state.vmin)
    with c3:
        vmax_input = st.number_input("vmax", value=st.session_state.vmax)
    with c4:
        st.write("")
        if st.button("Auto Scale (95%)"):
            finite_vals = display_map[np.isfinite(display_map)]
            if finite_vals.size > 0:
                vmax_auto = np.percentile(np.abs(finite_vals), 95)
                if vmax_auto <= 0: 
                    vmax_auto = 1.0
                st.session_state.vmin = -vmax_auto
                st.session_state.vmax = vmax_auto
                st.rerun()

    # Apply symmetric adjustment
    if vmin_input != st.session_state.vmin:
        st.session_state.vmin = vmin_input
        st.session_state.vmax = -vmin_input
    elif vmax_input != st.session_state.vmax:
        st.session_state.vmax = vmax_input
        st.session_state.vmin = -vmax_input
    
    # Ensure vmin < vmax
    if st.session_state.vmin >= st.session_state.vmax:
        st.session_state.vmin, st.session_state.vmax = (
            st.session_state.vmax, st.session_state.vmin
        )

    # --- PLOTTING ---
    mask_img = masker.mask_img_
    img_to_plot = masker.inverse_transform(display_map.astype(np.float32))

    cut_coords = None
    if nslices > 0:
        affine = mask_img.affine
        z_bounds = (
            affine[2, 3], 
            affine[2, 3] + affine[2, 2] * mask_img.shape[2]
        )
        cut_coords = np.linspace(z_bounds[0], z_bounds[1], nslices + 2)[1:-1]

    fig = plt.figure(figsize=(12, 5))
    plot_stat_map(
        img_to_plot,
        bg_img=None,
        display_mode='z',
        cut_coords=cut_coords,
        cmap='cold_hot',
        vmin=st.session_state.vmin,
        vmax=st.session_state.vmax,
        title=title_str,
        figure=fig,
        symmetric_cbar=True
    )
    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
