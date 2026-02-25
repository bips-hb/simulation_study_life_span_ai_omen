"""
utils.py  -  EHR Prediction & XAI Benchmark
============================================
Single utility module covering:
  1.  Reproducibility
  2.  Model definitions  (GRU, LSTM, LogitWrapper)
  3.  Data loading & splitting
  4.  Training & evaluation
  5.  Case-study selection
  6.  XAI - Integrated Gradients (IG)
  7.  XAI - Expected Gradients (EG)
  8.  Counterfactual explanations
  9.  Plotting helpers
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim

from captum.attr import IntegratedGradients
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# 1.  REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# ==============================================================================
# 2.  MODEL DEFINITIONS
# ==============================================================================

class GRUModel(nn.Module):
    """Gated Recurrent Unit classifier.

    Parameters
    ----------
    input_size   : number of features per timestep
    hidden_size  : GRU hidden dimension
    output_size  : number of output logits (1 for binary)
    dropout_rate : dropout probability applied before the FC layer
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, dropout_rate: float = 0.5):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class LSTMModel(nn.Module):
    """Long Short-Term Memory classifier.

    Parameters
    ----------
    input_size   : number of features per timestep
    hidden_size  : LSTM hidden dimension
    output_size  : number of output logits (1 for binary)
    dropout_rate : dropout probability applied before the FC layer
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, dropout_rate: float = 0.5):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class LogitWrapper(nn.Module):
    """Wraps any model so its output is always shape (B, 1).

    Required by SHAP GradientExplainer which expects a fixed output shape.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_logits(x)
        return out.unsqueeze(1) if out.dim() == 1 else out


# ==============================================================================
# 3.  DATA LOADING & SPLITTING
# ==============================================================================

def load_data(data_dir: str, n_patients: int, n_visits: int,
              scenario_tag: str = ""):
    """Load and assemble the multimodal EHR tensor for one simulation scenario.

    Parameters
    ----------
    data_dir     : folder produced by the R simulation (e.g. "sim_outputs")
    n_patients   : N_patients used when the scenario was simulated
    n_visits     : N_visits used when the scenario was simulated
    scenario_tag : tag string appended to every file by the R script,
                   e.g. "N3000_V20_trig0.15_esc0.10_bg0.010".
                   Pass "" (default) to load the original un-tagged files.

    Returns
    -------
    X             : np.ndarray  (N, T, F)
    y             : np.ndarray  (N,)
    tokens        : list[str]   sorted token names
    feature_names : list[str]   tokens + static column names
    labels        : pd.DataFrame phenotype ground-truth table
    """
    def _path(stem):
        fname = f"{stem}_{scenario_tag}.csv" if scenario_tag else f"{stem}.csv"
        return os.path.join(data_dir, fname)

    df_events = pd.read_csv(_path("ehr_events"))
    df_static = pd.read_csv(_path("static_data"))
    labels    = pd.read_csv(_path("phenotype_labels_gt"))

    tokens       = sorted(df_events["token"].unique())
    token_to_idx = {t: i for i, t in enumerate(tokens)}
    n_tokens     = len(tokens)

    X = np.zeros((n_patients, n_visits, n_tokens + 2), dtype=np.float32)

    for _, row in df_events.iterrows():
        X[int(row["patient_id"]) - 1,
          int(row["time"])       - 1,
          token_to_idx[row["token"]]] = 1.0

    for i in range(n_patients):
        X[i, :, n_tokens]     = df_static.loc[i, "static_bin"]
        X[i, :, n_tokens + 1] = df_static.loc[i, "static_cont"]

    y             = df_static["outcome"].values
    feature_names = list(tokens) + ["Static_Bin", "Static_Cont"]

    print(f"Loaded: X={X.shape}  |  prevalence={y.mean():.3f}")
    return X, y, tokens, feature_names, labels


def split_data(X, y, labels, patient_ids,
               test_size: float = 0.2, random_state: int = 3105):
    """Stratified train/test split; attaches test-row indices to labels.

    Returns
    -------
    X_train, X_test, y_train, y_test,
    pid_train, pid_test,
    labels_test   : phenotype table filtered to test patients, with 'test_idx' column
    """
    (X_train, X_test,
     y_train, y_test,
     pid_train, pid_test) = train_test_split(
        X, y, patient_ids,
        test_size=test_size, random_state=random_state, stratify=y
    )

    pid_to_test_index = {pid: i for i, pid in enumerate(pid_test)}
    labels_test = labels[labels["patient_id"].isin(pid_test)].copy()
    labels_test["test_idx"] = labels_test["patient_id"].map(pid_to_test_index)

    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, pid_train, pid_test, labels_test


# ==============================================================================
# 4.  TRAINING & EVALUATION
# ==============================================================================

def run_training_pipeline(model, X_train, y_train, X_val, y_val,
                           save_path, epochs=40, batch_size=32, lr=0.001):
    """Train *model* and save the best checkpoint (by val AUC).

    Returns
    -------
    history : dict with keys 'train_loss', 'train_auc', 'val_auc'
    """
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(X_train),
                      torch.Tensor(y_train).view(-1, 1)),
        batch_size=batch_size, shuffle=True,
    )
    val_tensor   = torch.Tensor(X_val)
    criterion    = nn.BCEWithLogitsLoss()
    optimizer    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history      = {"train_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss                  = 0.0
        y_true_train, y_pred_train  = [], []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model.forward_logits(batch_x)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            epoch_loss   += loss.item()
            y_true_train.extend(batch_y.detach().cpu().numpy().ravel())
            y_pred_train.extend(probs.detach().cpu().numpy().ravel())

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(
                model.forward_logits(val_tensor)
            ).cpu().numpy().ravel()

        val_auc   = roc_auc_score(y_val,                    val_probs)
        train_auc = roc_auc_score(np.array(y_true_train),   np.array(y_pred_train))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d} | "
                  f"Loss {history['train_loss'][-1]:.4f} | "
                  f"Val AUC {val_auc:.4f}")

    return history


def train_all_models(model_registry: dict, X_train, y_train, X_test, y_test,
                     train_config: dict, out_dir: str) -> dict:
    """Train every model in *model_registry* and return ``{name: model}``.

    Parameters
    ----------
    model_registry : dict mapping name -> constructor(input_size) -> nn.Module
    train_config   : kwargs forwarded to run_training_pipeline
    out_dir        : directory where best weights (.pth) are saved
    """
    input_size = X_train.shape[-1]
    trained    = {}

    for name, constructor in model_registry.items():
        print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
        model     = constructor(input_size)
        save_path = os.path.join(out_dir, f"best_{name.lower()}.pth")

        history = run_training_pipeline(
            model, X_train, y_train, X_test, y_test,
            save_path=save_path, **train_config,
        )
        plot_training_curves(history, name, out_dir)

        model.load_state_dict(torch.load(save_path))
        model.eval()
        trained[name] = model

    return trained


def plot_training_curves(history: dict, model_name: str, out_dir: str) -> None:
    """Save and display loss + AUC learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Loss")
    axes[0].set_title(f"{model_name} - Training Loss")
    axes[0].legend()

    axes[1].plot(history["train_auc"], label="Train AUC", linestyle="--")
    axes[1].plot(history["val_auc"],   label="Val AUC",   linewidth=2)
    axes[1].axhline(max(history["val_auc"]), color="r",
                    linestyle=":", label="Best Val")
    axes[1].set_title(f"{model_name} - AUC History")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"training_{model_name.lower()}.png"),
                dpi=150)
    plt.show()


def benchmark_table(trained_models: dict,
                    lr_model, rf_model,
                    X_train, X_test,
                    y_train, y_test) -> pd.DataFrame:
    """Build a Train/Test AUC summary table for all models (classic + DL)."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0],  -1)

    rows = [
        {"Model": "Logistic Regression",
         "Train AUC": roc_auc_score(y_train, lr_model.predict_proba(X_train_flat)[:, 1]),
         "Test AUC":  roc_auc_score(y_test,  lr_model.predict_proba(X_test_flat)[:, 1])},
        {"Model": "Random Forest",
         "Train AUC": roc_auc_score(y_train, rf_model.predict_proba(X_train_flat)[:, 1]),
         "Test AUC":  roc_auc_score(y_test,  rf_model.predict_proba(X_test_flat)[:, 1])},
    ]

    for name, model in trained_models.items():
        model.eval()
        with torch.no_grad():
            tr = model(torch.tensor(X_train, dtype=torch.float32)).numpy().ravel()
            te = model(torch.tensor(X_test,  dtype=torch.float32)).numpy().ravel()
        rows.append({"Model": name,
                     "Train AUC": roc_auc_score(y_train, tr),
                     "Test AUC":  roc_auc_score(y_test,  te)})

    df = pd.DataFrame(rows)
    df["Gap"] = (df["Train AUC"] - df["Test AUC"]).round(4)
    df[["Train AUC", "Test AUC"]] = df[["Train AUC", "Test AUC"]].round(4)
    return df.sort_values("Test AUC", ascending=False).reset_index(drop=True)


def evaluate_rnn(model_obj, weights_path: str, X, y) -> float:
    """Load weights and return AUC on (X, y)."""
    model_obj.load_state_dict(torch.load(weights_path))
    model_obj.eval()
    with torch.no_grad():
        preds = model_obj(torch.Tensor(X)).numpy()
    return roc_auc_score(y, preds)


# ==============================================================================
# 5.  CASE-STUDY SELECTION
# ==============================================================================

def select_case_studies(labels_test: pd.DataFrame,
                        y_test: np.ndarray,
                        y_probs_ref: np.ndarray,
                        phenotype_groups: list) -> dict:
    """Pick the highest-confidence true-positive patient per phenotype group.

    Parameters
    ----------
    phenotype_groups : list of (group_gt_value, display_title) tuples

    Returns
    -------
    dict mapping display_title -> test row index
    """
    def _pick_best(test_indices, probs):
        test_indices = np.array(test_indices, dtype=int)
        if len(test_indices) == 0:
            return None
        return int(test_indices[np.argmax(probs[test_indices])])

    case_studies = {}
    for group, title in phenotype_groups:
        cand = (labels_test
                .loc[labels_test["group_gt"] == group, "test_idx"]
                .dropna().astype(int).values)
        cand = cand[y_test[cand] == 1]
        p = _pick_best(cand, y_probs_ref)
        if p is not None:
            case_studies[title] = p

    fp = np.where((y_test == 0) & (y_probs_ref > 0.6))[0]
    p  = _pick_best(fp, y_probs_ref)
    if p is not None:
        case_studies["False Positive"] = p

    return case_studies


# ==============================================================================
# 6.  XAI – INTEGRATED GRADIENTS (IG)
# ==============================================================================

def run_ig_global(trained_models: dict,
                  X_test_tensor: torch.Tensor,
                  y_test: np.ndarray,
                  feature_names: list,
                  n_steps: int = 64,
                  out_dir: str = ".") -> dict:
    """Compute and plot population-level IG maps for every model.

    Returns
    -------
    global_maps : dict  {model_name: np.ndarray (T, F)}
    """
    T, F        = X_test_tensor.shape[1], len(feature_names)
    global_maps = {}

    for name, model in trained_models.items():
        probs      = model(X_test_tensor).detach().numpy().ravel()
        tp_indices = np.where((y_test == 1) & (probs > 0.5))[0]
        gmap       = np.zeros((T, F))
        ig         = IntegratedGradients(model.forward_logits)

        print(f"IG global - {name}  ({len(tp_indices)} true-positive patients)…")
        for idx in tp_indices:
            inp  = X_test_tensor[idx:idx+1].clone().detach().requires_grad_(True)
            attr = ig.attribute(inp, baselines=torch.zeros_like(inp),
                                target=0, n_steps=n_steps)
            gmap += attr.squeeze().cpu().detach().numpy()

        global_maps[name] = gmap / max(len(tp_indices), 1)

    _plot_ig_heatmaps(
        global_maps, feature_names, T,
        title_prefix="GLOBAL ATTRIBUTION (mean IG over true positives)",
        out_dir=out_dir, fname="ig_global_heatmap.png",
    )
    return global_maps


def run_ig_case_studies(trained_models: dict,
                        X_test_tensor: torch.Tensor,
                        y_test: np.ndarray,
                        feature_names: list,
                        case_studies: dict,
                        n_steps: int = 64,
                        out_dir: str = ".") -> None:
    """Compute and plot per-patient IG heatmaps for every case study."""
    T = X_test_tensor.shape[1]

    for title, p_idx in case_studies.items():
        attr_by_model = {}
        for name, model in trained_models.items():
            ig   = IntegratedGradients(model.forward_logits)
            inp  = X_test_tensor[p_idx:p_idx+1].clone().detach().requires_grad_(True)
            attr = ig.attribute(inp, baselines=torch.zeros_like(inp),
                                target=0, n_steps=n_steps)
            a = attr.squeeze().cpu().detach().numpy()
            attr_by_model[name] = a / (np.sum(np.abs(a)) + 1e-8)

        actual      = int(y_test[p_idx])
        model_probs = {n: m(X_test_tensor[p_idx:p_idx+1]).item()
                       for n, m in trained_models.items()}

        _plot_ig_heatmaps(
            attr_by_model, feature_names, T,
            title_prefix=f"IG  |  {title}  (Patient {p_idx}, y={actual})",
            model_probs=model_probs, out_dir=out_dir,
            fname=f"ig_case_{title.replace(' ', '_').replace(':', '').replace('+', '_')}.png",
        )


def _plot_ig_heatmaps(attr_dict: dict, feature_names: list, T: int,
                      title_prefix: str, model_probs: dict = None,
                      top_k: int = 15, out_dir: str = ".",
                      fname: str = "ig.png") -> None:
    """Internal: side-by-side heatmap for every model in attr_dict."""
    n_models    = len(attr_dict)
    shared_vmax = max(np.max(np.abs(a)) for a in attr_dict.values()) + 1e-12

    fig, axes = plt.subplots(1, n_models, figsize=(12 * n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (name, attr) in zip(axes, attr_dict.items()):
        top_idx  = np.argsort(np.abs(attr).sum(axis=0))[-top_k:][::-1]
        subtitle = (f"MODEL: {name}"
                    + (f"\nPred prob: {model_probs[name]:.4f}" if model_probs else ""))

        sns.heatmap(
            attr[:, top_idx].T,
            cmap="RdBu_r", center=0,
            vmin=-shared_vmax, vmax=shared_vmax, ax=ax,
            xticklabels=[f"V{t+1}" for t in range(T)],
            yticklabels=[feature_names[j] for j in top_idx],
            cbar_kws={"label": "Attribution Score"},
        )
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.set_xlabel("Visit")

    plt.suptitle(title_prefix, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# 7.  XAI – EXPECTED GRADIENTS (EG)
# ==============================================================================

def ensure_eg_NTF(shap_vals, T: int, F: int) -> np.ndarray:
    """Coerce EG output (from shap.GradientExplainer) into shape (N, T, F) regardless of SHAP version."""
    raw = np.array(shap_vals)
    if raw.ndim == 4 and raw.shape[1] == 1:   raw = raw[:, 0, :, :]
    if raw.ndim == 3 and raw.shape[1] == T:    return raw
    if raw.ndim == 2 and raw.shape[1] == T*F:  return raw.reshape(raw.shape[0], T, F)
    if raw.ndim == 4 and raw.shape[-1] == 1:   return raw[:, :, :, 0]
    raise ValueError(f"Unexpected EG output shape {raw.shape}")


def _scalar_base_value(base_vals) -> float:
    """Convert EG base value (from shap.GradientExplainer.expected_value — scalar / list / array) to a single float."""
    if base_vals is None:
        return 0.0
    try:
        return float(np.mean(base_vals))
    except Exception:
        return 0.0


def compute_eg_rnn(model, X_train_np, X_explain_np,
                   device=None, background_size=128,
                   nsamples=200, seed=42):
    """Run Expected Gradients (EG) via shap.GradientExplainer for a single model.

    Returns
    -------
    explainer : shap.GradientExplainer
    eg_ntf    : np.ndarray  (N, T, F)  EG attribution values
    base_vals : float  scalar base value
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = model.to(device).eval()
    rng    = np.random.default_rng(seed)
    idx_bg = rng.choice(len(X_train_np),
                        size=min(background_size, len(X_train_np)),
                        replace=False)

    background = torch.tensor(X_train_np[idx_bg], dtype=torch.float32, device=device)
    X_e        = torch.tensor(X_explain_np,        dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(model, background)
    raw_sv    = explainer.shap_values(X_e, nsamples=nsamples)
    if isinstance(raw_sv, list):
        raw_sv = raw_sv[0]

    base_vals = getattr(explainer, "expected_value", None)
    eg_ntf    = ensure_eg_NTF(raw_sv, T=X_explain_np.shape[1], F=X_explain_np.shape[2])
    return explainer, eg_ntf, base_vals


def run_eg_all_models(trained_models: dict,
                      X_train_np: np.ndarray,
                      X_test_np: np.ndarray,
                      eg_config: dict,
                      out_dir: str = "."):
    """Compute EG attributions for every model in trained_models.

    Returns
    -------
    eg_by_model   : dict  {name: np.ndarray (N, T, F)}  EG attribution values
    base_by_model : dict  {name: float}  scalar base values
    """
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eg_by_model   = {}
    base_by_model = {}

    for name, model in trained_models.items():
        print(f"EG – {name}…")
        wrapped = LogitWrapper(model)
        _, eg_ntf, base_val = compute_eg_rnn(
            wrapped, X_train_np, X_test_np, device=device, **eg_config
        )
        eg_by_model[name]   = eg_ntf
        base_by_model[name] = _scalar_base_value(base_val)
        print(f"  shape={eg_ntf.shape}  base_value={base_by_model[name]:.4f}")

    return eg_by_model, base_by_model


def global_importance_per_feature(eg_ntf: np.ndarray) -> np.ndarray:
    """Mean absolute EG value per feature, collapsed over patients and time."""
    return np.mean(np.abs(eg_ntf), axis=(0, 1))


def run_eg_global_plots(eg_by_model: dict,
                        X_test_np: np.ndarray,
                        feature_names: list,
                        out_dir: str = ".") -> None:
    """Global EG: bar chart + temporal line plot for every model."""
    for name, eg_ntf in eg_by_model.items():
        imp = global_importance_per_feature(eg_ntf)
        plot_global_bar(imp, feature_names,
                        title=f"{name}: Global Feature Importance (mean |EG|)",
                        top_k=15)
        plot_eg_over_time(eg_ntf, feature_names, top_k=8,
                          title=f"{name}: Mean |EG| across visits (top features)")
        plot_eg_beeswarm_all_visits(eg_ntf, X_test_np, feature_names, max_display=20)


def run_eg_local_plots(eg_by_model: dict,
                       base_by_model: dict,
                       X_test_np: np.ndarray,
                       feature_names: list,
                       patient_idx: int = 2,
                       out_dir: str = ".") -> None:
    """Local EG: per-visit waterfalls, aggregated waterfall, heatmap, cumulative."""
    for name, eg_ntf in eg_by_model.items():
        base_val = base_by_model[name]
        print(f"\n--- EG local plots | {name} | patient {patient_idx} ---")

        plot_patient_per_visit_waterfalls(
            eg_ntf, X_test_np, feature_names,
            patient_idx=patient_idx, base_value=base_val, top_k_visit=15)

        plot_patient_aggregated_waterfall(
            eg_ntf, X_test_np, feature_names,
            patient_idx=patient_idx, base_value=base_val, top_k_agg=15)

        plot_patient_signed_eg_heatmap(
            eg_ntf, feature_names,
            patient_idx=patient_idx, top_k=15)

        plot_patient_cumulative_prediction(
            eg_ntf, patient_idx=patient_idx,
            base_value=base_val, convert_to_probability=True)

        plot_feature_cumulative(
            eg_ntf, feature_names,
            patient_idx=patient_idx, feature_name="ATC_5")


def run_eg_case_studies(eg_by_model: dict,
                        feature_names: list,
                        case_studies: dict,
                        trained_models: dict,
                        X_test_tensor: torch.Tensor,
                        y_test: np.ndarray,
                        top_k: int = 15,
                        normalize: bool = False,
                        out_dir: str = ".") -> None:
    """Side-by-side EG heatmaps for every case study (all models)."""
    plot_eg_case_heatmaps_side_by_side(
        eg_by_model, feature_names, case_studies,
        trained_models, X_test_tensor, y_test,
        top_k=top_k, normalize=normalize,
    )


# ==============================================================================
# 8.  COUNTERFACTUAL EXPLANATIONS
# ==============================================================================

def _predict_proba_factory(model: nn.Module, device: torch.device):
    """Return a predict_proba(x_np) -> float closure bound to *model*."""
    model.eval()
    def _predict(x_np: np.ndarray) -> float:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            return float(model(x_t).reshape(-1).cpu().numpy()[0])
    return _predict


def _greedy_cf_remove(x: np.ndarray,
                      token_names: list,
                      predict_proba,
                      threshold: float = 0.5,
                      max_edits: int = 10,
                      mode: str = "token",
                      static_idx=None,
                      base_token_cost: float = 1.0,
                      base_timestep_cost: float = 5.0,
                      time_weight: float = 0.0,
                      allowed_timesteps=None):
    """Greedy cost-aware counterfactual search (remove evidence).

    Selects edits by benefit-per-cost  =  (p_current - p_candidate) / cost(edit).

    Parameters
    ----------
    x                  : (1, T, F) patient sequence
    token_names        : feature name list of length F
    predict_proba      : callable (1,T,F) -> float
    threshold          : target probability to drop below
    max_edits          : greedy budget
    mode               : 'token' (remove one feature) or 'timestep' (zero full visit)
    static_idx         : feature indices that must not be edited
    base_token_cost    : base cost of a single token removal
    base_timestep_cost : base cost of zeroing an entire timestep
    time_weight        : >0 penalises edits to earlier timesteps
    allowed_timesteps  : restrict edits to a subset of timestep indices
    """
    x_cf       = x.copy()
    T, F       = x_cf.shape[1], x_cf.shape[2]
    static_idx = set(static_idx or [])
    allowed_ts = list(allowed_timesteps) if allowed_timesteps is not None else list(range(T))
    p_cur      = predict_proba(x_cf)
    p_before   = p_cur
    edits      = []

    if p_cur < threshold:
        return x_cf, edits, p_before, p_cur

    for step in range(1, max_edits + 1):
        best, best_score, best_p = None, 0.0, p_cur

        if mode == "timestep":
            candidates = [(t, None) for t in allowed_ts]
        else:
            candidates = [(t, j)
                          for t in allowed_ts
                          for j in np.where(x_cf[0, t, :] > 0.5)[0]
                          if j not in static_idx]

        for t, j in candidates:
            x_try = x_cf.copy()
            if mode == "timestep":
                x_try[0, t, :] = 0.0
                for jj in static_idx:
                    x_try[0, t, jj] = x_cf[0, t, jj]
            else:
                x_try[0, t, j] = 0.0

            p_new = predict_proba(x_try)
            gain  = p_cur - p_new
            if gain <= 0:
                continue

            c = base_timestep_cost if mode == "timestep" else base_token_cost
            if time_weight > 0 and T > 1:
                c *= 1.0 + time_weight * (T - 1 - t) / (T - 1)
            score = gain / c

            if score > best_score:
                best_score, best_p, best = score, p_new, (t, j)

        if best is None:
            break

        t, j = best
        if mode == "timestep":
            x_cf[0, t, :] = 0.0
            for jj in static_idx:
                x_cf[0, t, jj] = x[0, t, jj]
            edits.append({"step": step, "type": "timestep_zero",
                           "t": t, "feature": None, "new_p": best_p})
        else:
            x_cf[0, t, j] = 0.0
            edits.append({"step": step, "type": "token_remove",
                           "t": t, "feature": token_names[j],
                           "j": j, "new_p": best_p})
        p_cur = best_p
        if p_cur < threshold:
            break

    return x_cf, edits, p_before, p_cur


def run_cf_case_studies(case_studies: dict,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        pid_test: np.ndarray,
                        feature_names: list,
                        tokens: list,
                        predict_proba,
                        threshold: float = 0.5,
                        max_edits: int = 8,
                        mode: str = "token",
                        time_weight: float = 0.0,
                        allowed_timesteps=None):
    """Run counterfactual search for every case in case_studies.

    Returns
    -------
    summary_df : pd.DataFrame  one row per case
    details    : dict  {case_title: pd.DataFrame of edits}
    """
    static_idx = [len(tokens), len(tokens) + 1]
    rows, details = [], {}

    for title, test_idx in case_studies.items():
        test_idx = int(test_idx)
        x = np.asarray(X_test[test_idx:test_idx+1], dtype=float)

        x_cf, edits, p_before, p_after = _greedy_cf_remove(
            x, feature_names, predict_proba,
            threshold=threshold, max_edits=max_edits, mode=mode,
            static_idx=static_idx, time_weight=time_weight,
            allowed_timesteps=allowed_timesteps,
        )
        details[title] = pd.DataFrame(edits)

        first_edit = None
        if edits:
            e = edits[0]
            first_edit = (f"{e['feature']} @ t={e['t']}"
                          if e.get("type") == "token_remove"
                          else f"timestep t={e['t']}")

        rows.append({"Case":       title,
                     "test_idx":   test_idx,
                     "patient_id": int(pid_test[test_idx]),
                     "y_true":     int(y_test[test_idx]),
                     "p_before":   float(p_before),
                     "p_after":    float(p_after),
                     "delta":      float(p_before - p_after),
                     "crossed":    bool(p_after < threshold),
                     "n_edits":    len(edits),
                     "first_edit": first_edit})

    return pd.DataFrame(rows).sort_values("Case").reset_index(drop=True), details


def run_cf_all_models(trained_models: dict,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      pid_test: np.ndarray,
                      feature_names: list,
                      tokens: list,
                      case_studies: dict,
                      cf_config: dict,
                      out_dir: str = ".") -> dict:
    """Run counterfactual search for every model; save CSV summaries.

    Returns
    -------
    dict  {model_name: {'summary': df, 'details': dict, 'edits_table': df}}
    """
    device     = next(next(iter(trained_models.values())).parameters()).device
    cf_results = {}

    for name, model in trained_models.items():
        print(f"\nCounterfactuals – {name}")
        predict_proba = _predict_proba_factory(model, device)

        summary, details = run_cf_case_studies(
            case_studies=case_studies,
            X_test=X_test, y_test=y_test, pid_test=pid_test,
            feature_names=feature_names, tokens=tokens,
            predict_proba=predict_proba, **cf_config,
        )
        edits_table = build_cf_edits_table(summary, details,
                                           threshold=cf_config.get("threshold", 0.5))

        summary.to_csv(
            os.path.join(out_dir, f"cf_summary_{name.lower()}.csv"), index=False)
        cf_results[name] = {"summary":     summary,
                             "details":     details,
                             "edits_table": edits_table}
        print(summary.to_string(index=False))

    return cf_results


def build_cf_edits_table(cf_summary: pd.DataFrame,
                          cf_details: dict,
                          threshold: float = 0.5) -> pd.DataFrame:
    """Flatten edit logs into a long DataFrame with step-level probability paths."""
    rows = []
    for _, r in cf_summary.iterrows():
        case     = r["Case"]
        p_prev   = float(r["p_before"])
        edits_df = cf_details.get(case, pd.DataFrame())

        if edits_df is None or edits_df.empty:
            rows.append({"Case":             case,
                          "patient_id":      int(r["patient_id"]),
                          "test_idx":        int(r["test_idx"]),
                          "y_true":          int(r["y_true"]),
                          "edit_step":       None,
                          "timestep":        None,
                          "feature":         None,
                          "p_before_step":   p_prev,
                          "p_after_step":    p_prev,
                          "delta_step":      0.0,
                          "crossed_this_step": False})
            continue

        for _, e in edits_df.iterrows():
            p_after = float(e["new_p"])
            delta   = p_prev - p_after
            crossed = (p_prev >= threshold) and (p_after < threshold)
            rows.append({"Case":             case,
                          "patient_id":      int(r["patient_id"]),
                          "test_idx":        int(r["test_idx"]),
                          "y_true":          int(r["y_true"]),
                          "edit_step":       int(e["step"]),
                          "timestep":        int(e["t"]) if pd.notnull(e.get("t")) else None,
                          "feature":         e.get("feature"),
                          "p_before_step":   p_prev,
                          "p_after_step":    p_after,
                          "delta_step":      delta,
                          "crossed_this_step": bool(crossed)})
            p_prev = p_after

    return (pd.DataFrame(rows)
              .sort_values(["Case", "patient_id", "edit_step"], na_position="last")
              .reset_index(drop=True))


def style_cf_table(df: pd.DataFrame):
    """Pandas Styler: alternating row colours + bold at threshold crossing."""
    case_codes = df["Case"].astype("category").cat.codes

    def _style(row):
        bg   = "#f5f5f5" if case_codes[row.name] % 2 == 0 else "#ffffff"
        bold = "; font-weight: bold" if row.get("crossed_this_step") else ""
        return [f"background-color: {bg}{bold}"] * len(row)

    return df.style.hide(axis="index").apply(_style, axis=1)


# ==============================================================================
# 9.  PLOTTING HELPERS  (EG)
# ==============================================================================

def plot_global_bar(imp: np.ndarray, feature_names: list,
                    title: str, top_k: int = 20) -> pd.DataFrame:
    """Horizontal bar chart of global mean |EG| per feature."""
    df = (pd.DataFrame({"feature": feature_names, "mean_abs_eg": imp})
            .sort_values("mean_abs_eg", ascending=False)
            .head(top_k)
            .iloc[::-1])

    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"], df["mean_abs_eg"])
    plt.title(title)
    plt.xlabel("Mean |EG|")
    plt.tight_layout()
    plt.show()
    return df.iloc[::-1]


def plot_eg_over_time(eg_ntf: np.ndarray, feature_names: list,
                      top_k: int = 8, title: str = "EG over time") -> None:
    """Line plot of mean |EG| per visit for the top-k features."""
    mean_abs   = np.mean(np.abs(eg_ntf), axis=0)
    global_imp = np.mean(np.abs(eg_ntf), axis=(0, 1))
    top_idx    = np.argsort(global_imp)[-top_k:]

    plt.figure(figsize=(10, 6))
    for j in top_idx:
        plt.plot(range(1, mean_abs.shape[0] + 1), mean_abs[:, j],
                 label=feature_names[j])
    plt.xlabel("Visit (1..T)")
    plt.ylabel("Mean |EG|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eg_beeswarm_all_visits(eg_ntf: np.ndarray, X_np: np.ndarray,
                                feature_names: list, max_display: int = 20) -> None:
    """EG beeswarm treating each (patient, visit) as one observation."""
    N, T, F  = eg_ntf.shape
    eg_flat  = eg_ntf.reshape(N * T, F)
    X_flat   = X_np.reshape(N * T, F)
    exp = shap.Explanation(values=eg_flat, data=X_flat,
                            feature_names=feature_names)
    shap.plots.beeswarm(exp, max_display=max_display)


def plot_patient_per_visit_waterfalls(eg_ntf, X_np, feature_names,
                                       patient_idx=2, base_value=0.0,
                                       top_k_visit=10, figsize=(10, 6)) -> None:
    """One EG waterfall plot per visit for a single patient."""
    T = eg_ntf.shape[1]
    for t in range(T):
        exp = shap.Explanation(
            values=eg_ntf[patient_idx, t, :],
            base_values=base_value,
            data=X_np[patient_idx, t, :],
            feature_names=feature_names,
        )
        plt.figure(figsize=figsize)
        shap.plots.waterfall(exp, max_display=top_k_visit, show=False)
        plt.title(f"Patient {patient_idx} | Visit {t+1}",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()


def plot_patient_aggregated_waterfall(eg_ntf, X_np, feature_names,
                                       patient_idx=2, base_value=0.0,
                                       top_k_agg=20, figsize=(12, 7)) -> None:
    """Waterfall of EG values summed across all visits."""
    exp = shap.Explanation(
        values=eg_ntf[patient_idx].sum(axis=0),
        base_values=base_value,
        data=X_np[patient_idx].sum(axis=0),
        feature_names=feature_names,
    )
    plt.figure(figsize=figsize)
    shap.plots.waterfall(exp, max_display=top_k_agg, show=False)
    plt.title(f"Patient {patient_idx} | Aggregated over all visits",
              fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_patient_signed_eg_heatmap(eg_ntf, feature_names,
                                    patient_idx=2, top_k=20,
                                    top_mode="mean_abs", vmax=None,
                                    figsize=(14, 8)) -> None:
    """Signed EG heatmap (features × visits) for one patient."""
    sv = eg_ntf[patient_idx]
    score_fns = {
        "mean_abs":   lambda s: np.mean(np.abs(s), axis=0),
        "sum_abs":    lambda s: np.sum( np.abs(s), axis=0),
        "sum_signed": lambda s: np.abs(np.sum(s,   axis=0)),
    }
    if top_mode not in score_fns:
        raise ValueError(f"top_mode must be one of {list(score_fns)}")

    top_idx = np.argsort(score_fns[top_mode](sv))[::-1][:top_k]
    mat     = sv[:, top_idx].T
    vmax    = vmax or (np.max(np.abs(mat)) + 1e-12)
    T       = eg_ntf.shape[1]

    plt.figure(figsize=figsize)
    sns.heatmap(
        mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=[f"V{t+1}" for t in range(T)],
        yticklabels=[feature_names[j] for j in top_idx],
        cbar_kws={"label": "EG value"},
    )
    plt.title(f"Patient {patient_idx} | Signed EG heatmap (top {top_k})",
              fontsize=16, fontweight="bold")
    plt.xlabel("Visit")
    plt.tight_layout()
    plt.show()


def plot_patient_cumulative_prediction(eg_ntf, patient_idx=2,
                                        base_value=0.0,
                                        convert_to_probability=True) -> None:
    """Cumulative predicted score/probability over visits for one patient."""
    T             = eg_ntf.shape[1]
    visit_contrib = eg_ntf[patient_idx].sum(axis=1)
    cumul_logit   = base_value + np.cumsum(visit_contrib)
    cumul         = expit(cumul_logit) if convert_to_probability else cumul_logit
    ylabel        = "Predicted probability" if convert_to_probability else "Logit"
    baseline_y    = expit(base_value) if convert_to_probability else base_value

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, T + 1), cumul, marker="o")
    plt.axhline(baseline_y, linestyle="--", color="gray", label="Baseline")
    plt.xlabel("Visit")
    plt.ylabel(ylabel)
    plt.title(f"Patient {patient_idx} | Cumulative prediction over visits",
              fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_cumulative(eg_ntf, feature_names, patient_idx,
                             feature_name: str, base_value: float = 0.0) -> None:
    """Cumulative EG contribution of a single feature over visits."""
    idx        = feature_names.index(feature_name)
    T          = eg_ntf.shape[1]
    cumulative = np.cumsum(eg_ntf[patient_idx, :, idx])

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, T + 1), cumulative, marker="o")
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(f"Cumulative EG over visits\nPatient {patient_idx} | {feature_name}",
              fontsize=13, fontweight="bold")
    plt.xlabel("Visit")
    plt.ylabel("Cumulative EG (logit)")
    plt.tight_layout()
    plt.show()


def plot_eg_case_heatmaps_side_by_side(eg_by_model: dict,
                                        feature_names: list,
                                        case_studies: dict,
                                        model_dict: dict,
                                        X_test_tensor: torch.Tensor,
                                        y_test: np.ndarray,
                                        top_k: int = 15,
                                        normalize: bool = False) -> None:
    """Side-by-side EG heatmaps for every case study, all models in one row."""
    n_models = len(model_dict)
    for title, p_idx in case_studies.items():
        fig, axes = plt.subplots(1, n_models,
                                  figsize=(12 * n_models, 8), sharey=False)
        if n_models == 1:
            axes = [axes]
        actual = int(y_test[p_idx])

        mats, vmaxs = {}, {}
        for name in model_dict:
            eg = eg_by_model[name][p_idx].copy()
            if normalize:
                eg /= (np.sum(np.abs(eg)) + 1e-8)
            mats[name]  = eg
            vmaxs[name] = np.max(np.abs(eg))

        shared_vmax = max(vmaxs.values())

        for ax, name in zip(axes, model_dict):
            eg      = mats[name]
            top_idx = np.argsort(np.sum(np.abs(eg), axis=0))[-top_k:][::-1]
            prob    = model_dict[name](X_test_tensor[p_idx:p_idx+1]).item()

            sns.heatmap(
                eg[:, top_idx].T,
                cmap="RdBu_r", center=0,
                vmin=-shared_vmax, vmax=shared_vmax, ax=ax,
                xticklabels=[f"V{t+1}" for t in range(eg.shape[0])],
                yticklabels=[feature_names[j] for j in top_idx],
                cbar_kws={"label": "EG value"},
            )
            ax.set_title(
                f"MODEL: {name}\nActual: {actual}  |  Pred: {prob:.4f}",
                fontsize=14, fontweight="bold",
            )

        plt.suptitle(f"{title}  (Patient {p_idx})",
                     fontsize=18, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.show()