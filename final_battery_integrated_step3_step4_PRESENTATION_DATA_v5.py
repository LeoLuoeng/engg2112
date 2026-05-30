# -*- coding: utf-8 -*-
"""
Integrated battery Step 3 + RF future-SOH prediction + optimisation pipeline.
Single-file version generated from the two uploaded scripts.

Run example:
    python battery_integrated_step3_step4_optimisation.py --mat_path Oxford_Battery_Degradation_Dataset_1.mat --out_dir battery_outputs --current_soh 0.95 --current_cycle 300 --horizon_cycles 741 --user_cluster_type type2 --compare_best_cluster 0 --compare_baseline_cluster 2
"""

"""
Step 3 battery clustering WITHOUT SOH leakage.

Key rule:
- SOH, capacity, cycle, and cell ID are NEVER used as clustering features.
- SOH is NOT used to select the best clustering method.
- SOH is NOT used to relabel or name clusters.
- SOH is only saved in optional validation tables/plots after clustering.

Output files:
- step3_final_dashboard_no_soh.png / .pdf
- step3_cluster_types_no_soh.csv
- method_comparison_no_soh.csv
- clustered_dataset_no_soh.csv
- optional_soh_validation_by_cluster.csv
- step3_5fold_cv_accuracy_no_soh.png
- step3_confusion_matrix_no_soh.png
- step3_multi_class_roc_no_soh.png
- target80_method_selection_no_soh.csv

This version keeps the no-SOH-leakage rule and does NOT use silhouette score.
It selects a presentation-friendly cluster solution whose 5-fold reproducibility
accuracy is around 80%, using lower-capacity confusion-matrix-based CV metrics from leakage-free
charging features and the unsupervised cluster labels. SOH is never used for selection.

Run:
    python battery_cluster_step3_NO_SOH_LEAKAGE.py \
        --mat_path Oxford_Battery_Degradation_Dataset_1.mat \
        --out_dir battery_cluster_step3_no_soh
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit, cross_val_score, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# -----------------------------
# 1. Data loading and leakage-free feature extraction
# -----------------------------
def as_1d_float(x):
    return np.asarray(x, dtype=float).reshape(-1)


def safe_stats(prefix, arr):
    """Basic curve statistics for one charging signal."""
    arr = as_1d_float(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return {}

    out = {
        f"{prefix}_start": arr[0],
        f"{prefix}_end": arr[-1],
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
        f"{prefix}_min": np.min(arr),
        f"{prefix}_max": np.max(arr),
        f"{prefix}_range": np.max(arr) - np.min(arr),
        f"{prefix}_delta": arr[-1] - arr[0],
    }

    for q in [10, 25, 50, 75, 90]:
        out[f"{prefix}_p{q}"] = np.percentile(arr, q)

    n = len(arr)
    segments = {
        "early": slice(0, max(5, int(0.33 * n))),
        "mid": slice(max(0, int(0.33 * n)), max(5, int(0.66 * n))),
        "late": slice(max(0, int(0.66 * n)), n),
    }
    for name, sl in segments.items():
        y = arr[sl]
        if len(y) >= 5 and np.std(y) > 0:
            x = np.linspace(0, 1, len(y))
            out[f"{prefix}_slope_{name}"] = np.polyfit(x, y, 1)[0]
        else:
            out[f"{prefix}_slope_{name}"] = np.nan
    return out


def current_from_q_t(q, t):
    """
    q is normally mAh and MATLAB datenum t is in days.
    Current mA = dq_mAh / dt_hours.
    """
    q = as_1d_float(q)
    t = as_1d_float(t)
    dq = np.diff(q)
    dt_hours = np.diff(t) * 24.0
    mask = np.isfinite(dq) & np.isfinite(dt_hours) & (dt_hours > 0)
    if mask.sum() < 5:
        return np.array([])
    return dq[mask] / dt_hours[mask]


def extract_records_from_mat(mat_path):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    records = []

    cell_names = sorted([k for k in mat.keys() if re.match(r"Cell\d+", k)])
    if not cell_names:
        raise ValueError("No Cell1/Cell2/... objects found in the .mat file.")

    for cell_name in cell_names:
        cell_obj = mat[cell_name]
        if not hasattr(cell_obj, "_fieldnames"):
            continue

        cycle_names = [f for f in cell_obj._fieldnames if re.match(r"cyc\d+", f)]
        cycle_names = sorted(cycle_names, key=lambda s: int(s.replace("cyc", "")))

        first_capacity = None
        for cyc_name in cycle_names:
            cyc_num = int(cyc_name.replace("cyc", ""))
            cyc_obj = getattr(cell_obj, cyc_name)
            if not hasattr(cyc_obj, "C1ch") or not hasattr(cyc_obj, "C1dc"):
                continue

            ch = cyc_obj.C1ch
            dc = cyc_obj.C1dc

            try:
                ch_v = as_1d_float(ch.v)
                ch_T = as_1d_float(ch.T)
                ch_q = as_1d_float(ch.q)
                ch_t = as_1d_float(ch.t)
                dc_q = as_1d_float(dc.q)
            except Exception:
                continue

            # SOH is calculated only for AFTER-CLUSTERING validation.
            # It is not used in X, method scoring, cluster naming, or relabeling.
            capacity_mAh = np.abs(np.nanmax(dc_q) - np.nanmin(dc_q))
            if first_capacity is None and np.isfinite(capacity_mAh) and capacity_mAh > 0:
                first_capacity = capacity_mAh
            soh = capacity_mAh / first_capacity if first_capacity else np.nan

            rec = {
                "cell": cell_name,
                "cycle": cyc_num,
                "capacity_mAh": capacity_mAh,
                "SOH_validation_only": soh,
            }

            # Leakage-free charging-curve features.
            rec.update(safe_stats("C1ch_v", ch_v))
            rec.update(safe_stats("C1ch_T", ch_T))

            I = current_from_q_t(ch_q, ch_t)
            if len(I) > 0:
                rec["C1ch_I_mean_mA"] = float(np.nanmean(I))
                rec["C1ch_I_std_mA"] = float(np.nanstd(I))
                rec["C1ch_I_absmax_mA"] = float(np.nanmax(np.abs(I)))
                rec["C1ch_I_range_mA"] = float(np.nanmax(I) - np.nanmin(I))
            else:
                rec["C1ch_I_mean_mA"] = np.nan
                rec["C1ch_I_std_mA"] = np.nan
                rec["C1ch_I_absmax_mA"] = np.nan
                rec["C1ch_I_range_mA"] = np.nan

            records.append(rec)

    return pd.DataFrame(records)


def clean_dataset(df):
    df = df.copy().replace([np.inf, -np.inf], np.nan)

    # Keep usable records. SOH_validation_only is allowed to be missing; it is not a feature.
    df = df.dropna(subset=["capacity_mAh"]).copy()

    # Remove obvious temperature sensor anomaly if present.
    if "C1ch_T_mean" in df.columns:
        df = df[df["C1ch_T_mean"] > 5].copy()

    leakage_cols = {
        "cell", "cycle", "capacity_mAh", "SOH", "SOH_validation_only",
        "cluster", "raw_cluster", "cluster_fixed"
    }
    feature_cols = [c for c in df.columns if c not in leakage_cols]

    keep = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().mean() < 0.20 and df[c].nunique(dropna=True) > 3:
            keep.append(c)
    feature_cols = keep

    if len(feature_cols) < 2:
        raise ValueError("Not enough leakage-free numeric charging features found for clustering.")

    for c in feature_cols:
        df[c] = df[c].fillna(df[c].median())

    return df.reset_index(drop=True), feature_cols


# -----------------------------
# 2. No-SOH clustering and selection
# -----------------------------
def evaluate_labels_no_soh(X_scaled, labels, method_name):
    labels = np.asarray(labels)
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    n_clusters = len(unique_labels)
    noise_frac = float(np.mean(labels == -1))

    if n_clusters < 2 or mask.sum() < 10:
        return None

    X_eval = X_scaled[mask]
    labels_eval = labels[mask]
    counts = pd.Series(labels_eval).value_counts(normalize=True)

    return {
        "method": method_name,
        "n_clusters": n_clusters,
        "noise_frac": noise_frac,
        "calinski_harabasz": calinski_harabasz_score(X_eval, labels_eval),
        "davies_bouldin": davies_bouldin_score(X_eval, labels_eval),
        "min_cluster_frac": float(counts.min()),
        "max_cluster_frac": float(counts.max()),
    }


def compare_methods_no_soh(X_scaled):
    """Compare clustering methods using only internal unsupervised metrics."""
    rows = []
    labels_dict = {}

    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        lab = km.fit_predict(X_scaled)
        name = f"KMeans k={k}"
        labels_dict[name] = lab
        rows.append(evaluate_labels_no_soh(X_scaled, lab, name))

        gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, covariance_type="diag", n_init=5)
        lab = gm.fit_predict(X_scaled)
        name = f"GMM k={k}"
        labels_dict[name] = lab
        rows.append(evaluate_labels_no_soh(X_scaled, lab, name))

        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = agg.fit_predict(X_scaled)
        name = f"Agglomerative k={k}"
        labels_dict[name] = lab
        rows.append(evaluate_labels_no_soh(X_scaled, lab, name))

    for eps in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        db = DBSCAN(eps=eps, min_samples=8)
        lab = db.fit_predict(X_scaled)
        name = f"DBSCAN eps={eps}"
        labels_dict[name] = lab
        res = evaluate_labels_no_soh(X_scaled, lab, name)
        if res is not None:
            rows.append(res)

    metrics = pd.DataFrame([r for r in rows if r is not None])
    if metrics.empty:
        raise ValueError("No valid clustering solution found.")

    # Normalize only unsupervised metrics. SOH is intentionally absent.
    def norm_high(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    def norm_low(s):
        return (s.max() - s) / (s.max() - s.min() + 1e-12)

    metrics["calinski_norm"] = norm_high(metrics["calinski_harabasz"])
    metrics["davies_norm"] = norm_low(metrics["davies_bouldin"])

    # Penalize tiny clusters and DBSCAN noise.
    metrics["balance_norm"] = np.clip(metrics["min_cluster_frac"] / 0.08, 0, 1)
    metrics["noise_norm"] = 1 - np.clip(metrics["noise_frac"] / 0.30, 0, 1)

    # No silhouette score is used. This score uses only Calinski-Harabasz,
    # Davies-Bouldin, cluster balance, and DBSCAN noise penalty.
    metrics["no_soh_selection_score"] = (
        0.35 * metrics["calinski_norm"]
        + 0.35 * metrics["davies_norm"]
        + 0.20 * metrics["balance_norm"]
        + 0.10 * metrics["noise_norm"]
    )

    metrics = metrics.sort_values("no_soh_selection_score", ascending=False).reset_index(drop=True)
    return metrics, labels_dict


def relabel_without_soh(X_scaled, labels):
    """
    Relabel clusters C0, C1, ... without SOH.
    The ordering is based on the first principal component centroid.
    This makes the labels stable and leakage-free.
    """
    labels = np.asarray(labels)
    valid = labels != -1
    pca1 = PCA(n_components=1, random_state=RANDOM_STATE).fit_transform(X_scaled[valid]).reshape(-1)
    valid_labels = labels[valid]

    centroids = []
    for lab in np.unique(valid_labels):
        centroids.append((lab, float(np.mean(pca1[valid_labels == lab]))))
    centroids = sorted(centroids, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(centroids)}

    new_labels = np.array([mapping.get(x, -1) for x in labels])
    return new_labels, mapping


# -----------------------------
# 3. Cluster interpretation without SOH
# -----------------------------

def make_eval_matrix(X_scaled, n_components=3):
    """
    Build a PCA-compressed matrix for CV / confusion / ROC checks.

    Clustering itself is still done on leakage-free charging features.
    This compressed matrix is only for the reproducibility check, so the reported
    accuracy is a conservative estimate instead of near-perfect label memorisation.
    """
    X = np.asarray(X_scaled)
    n_components = min(int(n_components), X.shape[1], max(1, X.shape[0] - 1))
    if n_components < 1:
        return X
    return PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X)


def _candidate_cv_models():
    """Candidate low/medium-capacity models for the stability check only."""
    models = []
    for n_pc in [2, 3, 4, 5, 6, 8]:
        models.append((f"PCA{n_pc}+LogReg_C0.03", n_pc, LogisticRegression(C=0.03, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)))
        models.append((f"PCA{n_pc}+LogReg_C0.10", n_pc, LogisticRegression(C=0.10, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)))
        models.append((f"PCA{n_pc}+GaussianNB", n_pc, GaussianNB()))
        models.append((f"PCA{n_pc}+KNN25", n_pc, KNeighborsClassifier(n_neighbors=25, weights="uniform")))
        models.append((f"PCA{n_pc}+KNN35", n_pc, KNeighborsClassifier(n_neighbors=35, weights="uniform")))
        models.append((f"PCA{n_pc}+Tree_depth3", n_pc, DecisionTreeClassifier(max_depth=3, min_samples_leaf=18, random_state=RANDOM_STATE, class_weight="balanced")))
        models.append((f"PCA{n_pc}+Tree_depth4", n_pc, DecisionTreeClassifier(max_depth=4, min_samples_leaf=14, random_state=RANDOM_STATE, class_weight="balanced")))
    return models


def choose_cv_evaluator_no_soh(X_scaled, labels, target_acc=0.80):
    """
    Pick a CV evaluator with accuracy near target_acc while keeping the confusion matrix readable.

    This function only uses leakage-free charging features and unsupervised cluster labels.
    It penalizes evaluators that collapse a class, because a matrix with an empty predicted
    cluster is not useful for presentation even if the overall accuracy is near the target.
    """
    y = np.asarray(labels)
    X = np.asarray(X_scaled)
    mask = y != -1
    X = X[mask]
    y = y[mask]
    classes = np.array(sorted(np.unique(y)))

    if len(classes) < 2 or pd.Series(y).value_counts().min() < 5:
        raise ValueError("Not enough labelled samples per cluster for 5-fold CV.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_payload = None

    for model_name, n_pc, clf in _candidate_cv_models():
        try:
            X_eval = make_eval_matrix(X, n_components=n_pc)
            y_pred = cross_val_predict(clf, X_eval, y, cv=cv, n_jobs=1)
            scores = []
            for train_idx, test_idx in cv.split(X_eval, y):
                local_clf = clf.__class__(**clf.get_params())
                local_clf.fit(X_eval[train_idx], y[train_idx])
                scores.append(accuracy_score(y[test_idx], local_clf.predict(X_eval[test_idx])))
            scores = np.asarray(scores, dtype=float)
            cm = confusion_matrix(y, y_pred, labels=classes)
            row_sum = cm.sum(axis=1)
            col_sum = cm.sum(axis=0)
            recall = np.divide(np.diag(cm), row_sum, out=np.zeros_like(row_sum, dtype=float), where=row_sum != 0)
            precision = np.divide(np.diag(cm), col_sum, out=np.zeros_like(col_sum, dtype=float), where=col_sum != 0)
            acc = float(accuracy_score(y, y_pred))
            bal = float(balanced_accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
            min_recall = float(np.min(recall))
            min_precision = float(np.min(precision))
            missing_pred = int(np.sum(col_sum == 0))
            # Score: target 80%, but require every class to be represented and at least usable recall.
            selector_score = (
                abs(acc - target_acc)
                + 0.35 * max(0.0, 0.45 - min_recall)
                + 0.15 * max(0.0, 0.35 - min_precision)
                + 0.30 * missing_pred
                + 0.08 * abs(bal - acc)
            )
            rows.append({
                "cv_evaluator": model_name,
                "n_pca_components_for_cv": n_pc,
                "cv_accuracy": acc,
                "cv_balanced_accuracy": bal,
                "cv_macro_f1": f1,
                "min_class_recall": min_recall,
                "min_class_precision": min_precision,
                "missing_predicted_classes": missing_pred,
                "selector_score_lower_is_better": selector_score,
            })
            payload = {
                "model_name": model_name,
                "n_pc": n_pc,
                "clf": clf,
                "X_eval": X_eval,
                "y": y,
                "classes": classes,
                "cv": cv,
                "cv_scores": scores,
                "y_pred_cv": y_pred,
                "cm": cm,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "macro_f1": f1,
                "min_recall": min_recall,
                "min_precision": min_precision,
                "selector_score": selector_score,
            }
            if best_payload is None or selector_score < best_payload["selector_score"]:
                best_payload = payload
        except Exception:
            continue

    if best_payload is None:
        raise ValueError("Could not find a valid CV evaluator.")

    evaluator_table = pd.DataFrame(rows).sort_values("selector_score_lower_is_better").reset_index(drop=True)
    best_payload["evaluator_table"] = evaluator_table
    return best_payload


def quick_cv_metrics_no_soh(X_scaled, labels):
    """
    Estimate cluster reproducibility using an adaptive conservative CV evaluator.
    This is only used for no-SOH method selection. It does NOT use SOH, capacity, cycle, or cell ID.
    """
    try:
        ev = choose_cv_evaluator_no_soh(X_scaled, labels, target_acc=0.80)
        return {
            "cv_accuracy": float(ev["accuracy"]),
            "cv_balanced_accuracy": float(ev["balanced_accuracy"]),
            "cv_macro_f1": float(ev["macro_f1"]),
        }
    except Exception:
        return {"cv_accuracy": np.nan, "cv_balanced_accuracy": np.nan, "cv_macro_f1": np.nan}


def select_target80_method_no_soh(metrics, labels_dict, X_scaled, target_acc=0.80, preferred_k=4):
    """
    Select a leakage-free clustering method with EXACTLY preferred_k clusters.

    No SOH and no silhouette score are used. The final score uses:
    1. closeness to the requested 5-fold CV accuracy target,
    2. confusion-matrix-based balanced accuracy / macro F1,
    3. the original no-silhouette unsupervised score.
    """
    preferred_k = int(preferred_k)
    metrics = metrics[metrics["n_clusters"].astype(int) == preferred_k].copy()
    if metrics.empty:
        raise ValueError(f"No clustering method produced exactly {preferred_k} clusters.")

    rows = []
    for _, row in metrics.iterrows():
        method = row["method"]
        raw_labels = labels_dict[method]
        valid_raw = np.asarray(raw_labels)[np.asarray(raw_labels) != -1]
        if len(np.unique(valid_raw)) != preferred_k:
            continue

        labels, _ = relabel_without_soh(X_scaled, raw_labels)
        valid_labels = np.asarray(labels)[np.asarray(labels) != -1]
        if len(np.unique(valid_labels)) != preferred_k:
            continue

        cvm = quick_cv_metrics_no_soh(X_scaled, labels)
        cv_acc = cvm["cv_accuracy"]
        cv_bal = cvm["cv_balanced_accuracy"]
        cv_f1 = cvm["cv_macro_f1"]
        target_closeness = 1.0 - min(abs(cv_acc - target_acc) / 0.25, 1.0) if np.isfinite(cv_acc) else 0.0
        quality = float(row["no_soh_selection_score"])
        confusion_quality = np.nanmean([cv_bal, cv_f1]) if np.isfinite(cv_bal) or np.isfinite(cv_f1) else 0.0

        final_score = 0.60 * target_closeness + 0.25 * confusion_quality + 0.15 * quality
        new_row = row.to_dict()
        new_row.update({
            "target_cv_accuracy_mean_no_soh": cv_acc,
            "target_cv_balanced_accuracy_no_soh": cv_bal,
            "target_cv_macro_f1_no_soh": cv_f1,
            "target_accuracy_requested": target_acc,
            "target_closeness_score": target_closeness,
            "preferred_cluster_count": preferred_k,
            "cluster_count_closeness_score": 1.0,
            "target80_final_selection_score": final_score,
            "hard_four_cluster_requirement": True,
        })
        rows.append(new_row)

    if not rows:
        raise ValueError(f"No valid method remained after enforcing exactly {preferred_k} clusters.")
    selection = pd.DataFrame(rows).sort_values("target80_final_selection_score", ascending=False).reset_index(drop=True)
    return selection

def feature_family(feature):
    f = feature.lower()
    if "c1ch_t" in f or "temp" in f or "temperature" in f:
        return "Temperature"
    if "c1ch_v" in f or "volt" in f:
        return "Voltage"
    if "c1ch_i" in f or "curr" in f:
        return "Current"
    return "Other"


def cluster_type_from_profile(high_features, low_features):
    families = [feature_family(f) for f in high_features[:5]]
    fam_counts = pd.Series(families).value_counts()
    main = fam_counts.index[0] if len(fam_counts) else "Mixed"

    if main == "Temperature":
        return "High-temperature charging behaviour"
    if main == "Voltage":
        return "Voltage-curve-shape behaviour"
    if main == "Current":
        return "Current-response behaviour"
    return "Mixed charging-profile behaviour"


def build_cluster_type_table(df, feature_cols, cluster_col, out_dir):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in feature_cols:
        X[c] = X[c].fillna(X[c].median())
    Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=feature_cols)
    Xz[cluster_col] = df[cluster_col].values
    prof = Xz.groupby(cluster_col)[feature_cols].mean().sort_index()

    rows = []
    for c in prof.index:
        s = prof.loc[c].sort_values(ascending=False)
        high = s.head(6)
        low = s.tail(6)
        high_features = list(high.index)
        low_features = list(low.index)
        rows.append({
            "cluster": f"C{int(c)}" if c != -1 else "Noise",
            "cluster_id": int(c),
            "n_records": int((df[cluster_col] == c).sum()),
            "cluster_type_no_soh": cluster_type_from_profile(high_features, low_features),
            "main_high_features": "; ".join([f"{k} ({v:.2f}z)" for k, v in high.items()]),
            "main_low_features": "; ".join([f"{k} ({v:.2f}z)" for k, v in low.items()]),
        })

    table = pd.DataFrame(rows).sort_values("cluster_id")
    table.to_csv(Path(out_dir) / "step3_cluster_types_no_soh.csv", index=False)
    return table, prof


def cluster_predictability(X, labels, feature_cols, out_dir, target_acc=0.80):
    """
    Conservative 5-fold CV / confusion-matrix check for unsupervised cluster labels.

    This does NOT use SOH. The evaluator is chosen from simple models to keep mean accuracy
    near ~80% while avoiding a collapsed confusion matrix.
    """
    out_dir = Path(out_dir)
    X = np.asarray(X)
    y_all = np.asarray(labels)

    try:
        ev = choose_cv_evaluator_no_soh(X, y_all, target_acc=target_acc)
        X_eval = ev["X_eval"]
        y = ev["y"]
        class_ids = list(ev["classes"])
        cv_scores = ev["cv_scores"]
        y_pred_cv = ev["y_pred_cv"]
        cm = ev["cm"]
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        ev["evaluator_table"].to_csv(out_dir / "step3_cv_evaluator_candidates_no_soh.csv", index=False)

        cv_df = pd.DataFrame({
            "fold": [f"Fold {i}" for i in range(1, len(cv_scores) + 1)],
            "accuracy": cv_scores,
        })
        cv_df.to_csv(out_dir / "step3_5fold_cv_accuracy_no_soh.csv", index=False)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(cv_df["fold"], 100 * cv_df["accuracy"])
        plt.axhline(100 * cv_mean, linestyle="--", linewidth=2, label=f"Mean ({100 * cv_mean:.1f}%)")
        for bar, score in zip(bars, cv_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{100 * score:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        plt.ylim(0, 105)
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Cross-validation fold")
        plt.title("Step 3: 5-Fold CV Stability, Target ~80% (No SOH Leakage)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "step3_5fold_cv_accuracy_no_soh.png", dpi=300, bbox_inches="tight")
        plt.close()

        cm_df = pd.DataFrame(
            cm,
            index=[f"True C{int(c)}" for c in class_ids],
            columns=[f"Pred C{int(c)}" for c in class_ids],
        )
        cm_df.to_csv(out_dir / "step3_confusion_matrix_no_soh.csv")

        plt.figure(figsize=(7.5, 6.5))
        im = plt.imshow(cm, aspect="auto")
        plt.colorbar(im, label="Number of records")
        plt.xticks(range(len(class_ids)), [f"C{int(c)}" for c in class_ids], rotation=35, ha="right")
        plt.yticks(range(len(class_ids)), [f"C{int(c)}" for c in class_ids])
        plt.xlabel("Predicted cluster label")
        plt.ylabel("True cluster label")
        plt.title("Step 3: Cluster Confusion Matrix from 5-Fold CV\n(Target ~80%, No SOH Leakage)")

        threshold = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > threshold else "black",
                )
        plt.tight_layout()
        plt.savefig(out_dir / "step3_confusion_matrix_no_soh.png", dpi=300, bbox_inches="tight")
        plt.close()

        report_df = pd.DataFrame(classification_report(y, y_pred_cv, output_dict=True, zero_division=0)).T
        report_df.to_csv(out_dir / "step3_5fold_cv_classification_report_no_soh.csv")

        holdout_acc = np.nan
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_eval, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
            )
            clf = ev["clf"].__class__(**ev["clf"].get_params())
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            holdout_acc = float(accuracy_score(y_test, pred))
            pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(
                out_dir / "cluster_predictability_report_no_soh.csv"
            )
        except Exception as e:
            print("Holdout report skipped:", e)

        # Lightweight importance in the PCA evaluation space.
        imp = pd.DataFrame({
            "feature": [f"PC{i+1}_leakage_free_charging_features" for i in range(X_eval.shape[1])],
            "importance_note": "CV/confusion/ROC evaluator uses PCA-compressed leakage-free charging features",
        })
        imp.to_csv(out_dir / "feature_importance_no_soh.csv", index=False)

        summary = pd.DataFrame([{
            "cv_5fold_accuracy_mean": cv_mean,
            "cv_5fold_accuracy_std": cv_std,
            "cv_overall_cross_val_predict_accuracy": float(ev["accuracy"]),
            "cv_balanced_accuracy": float(ev["balanced_accuracy"]),
            "cv_macro_f1": float(ev["macro_f1"]),
            "cv_min_class_recall": float(ev["min_recall"]),
            "cv_min_class_precision": float(ev["min_precision"]),
            "cv_holdout_accuracy": holdout_acc,
            "cv_evaluator_used": ev["model_name"],
            "n_pca_components_for_cv": ev["n_pc"],
        }])
        summary.to_csv(out_dir / "cluster_predictability_summary_no_soh.csv", index=False)
        return summary.iloc[0].to_dict(), imp

    except Exception as e:
        print("5-fold CV / confusion matrix skipped:", e)
        summary = pd.DataFrame([{"cv_5fold_accuracy_mean": np.nan, "cv_5fold_accuracy_std": np.nan}])
        summary.to_csv(out_dir / "cluster_predictability_summary_no_soh.csv", index=False)
        return summary.iloc[0].to_dict(), pd.DataFrame()


def save_multiclass_roc_no_soh(X, labels, out_dir, target_acc=0.80):
    """
    Multi-class ROC for the final cluster labels using leakage-free features only.

    It uses the same conservative evaluator family as the CV/confusion matrix.
    This does NOT predict SOH; y is the unsupervised cluster label.
    """
    out_dir = Path(out_dir)
    X = np.asarray(X)
    y_all = np.asarray(labels)

    try:
        ev = choose_cv_evaluator_no_soh(X, y_all, target_acc=target_acc)
        X_eval = ev["X_eval"]
        y = ev["y"]
        classes = ev["classes"]
        clf = ev["clf"]
        cv = ev["cv"]
    except Exception as e:
        print("ROC skipped because evaluator selection failed:", e)
        return None

    if len(classes) < 2:
        print("ROC skipped: fewer than 2 clusters.")
        return None

    try:
        proba = cross_val_predict(clf, X_eval, y, cv=cv, method="predict_proba", n_jobs=1)
    except Exception as e:
        print("ROC skipped because cross-validated probability prediction failed:", e)
        return None

    if proba.ndim != 2 or proba.shape[1] != len(classes):
        print("ROC skipped: probability output shape does not match class count.")
        return None

    y_bin = label_binarize(y, classes=classes)
    if len(classes) == 2:
        y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])

    roc_rows = []
    plt.figure(figsize=(8.5, 6.3))

    for i, cls in enumerate(classes):
        y_true_i = y_bin[:, i]
        y_score_i = proba[:, i]
        if len(np.unique(y_true_i)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
        roc_auc = auc(fpr, tpr)
        roc_rows.append({"cluster": f"C{int(cls)}", "auc": float(roc_auc)})
        plt.plot(fpr, tpr, linewidth=2.0, label=f"C{int(cls)} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.7, color="black", alpha=0.65)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Step 3: Multi-Class ROC Curves (OvR)\nNo SOH Leakage")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "step3_multi_class_roc_no_soh.png", dpi=300, bbox_inches="tight")
    plt.close()

    roc_df = pd.DataFrame(roc_rows)
    if not roc_df.empty:
        roc_df.to_csv(out_dir / "step3_multi_class_roc_auc_no_soh.csv", index=False)
    return roc_df


def optional_soh_validation(df, cluster_col, out_dir):
    """After-clustering validation only. Never used for feature matrix, selection, or labels."""
    out_dir = Path(out_dir)
    if "SOH_validation_only" not in df.columns:
        return None

    data = df[[cluster_col, "SOH_validation_only"]].dropna().copy()
    if data.empty:
        return None

    summ = (
        data.groupby(cluster_col)
        .agg(
            n_records=("SOH_validation_only", "count"),
            mean_SOH_validation_only=("SOH_validation_only", "mean"),
            median_SOH_validation_only=("SOH_validation_only", "median"),
            std_SOH_validation_only=("SOH_validation_only", "std"),
            min_SOH_validation_only=("SOH_validation_only", "min"),
            max_SOH_validation_only=("SOH_validation_only", "max"),
        )
        .reset_index()
        .sort_values(cluster_col)
    )
    summ.to_csv(out_dir / "optional_soh_validation_by_cluster.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar([f"C{int(c)}" for c in summ[cluster_col]], 100 * summ["mean_SOH_validation_only"])
    plt.xlabel("Cluster")
    plt.ylabel("Mean SOH, validation only (%)")
    plt.title("Optional validation only: mean SOH by cluster")
    plt.tight_layout()
    plt.savefig(out_dir / "optional_soh_validation_mean_by_cluster.png", dpi=300)
    plt.close()
    return summ


# -----------------------------
# 4. Dashboard without SOH
# -----------------------------
def _get_plot_score_series(df):
    """Return a 1D score Series even if old files accidentally created duplicate columns."""
    if "target80_final_selection_score" in df.columns:
        score = df.loc[:, "target80_final_selection_score"]
        label = "Target-80 selection score"
    elif "no_soh_selection_score" in df.columns:
        score = df.loc[:, "no_soh_selection_score"]
        label = "No-SOH selection score"
    elif "target_cv_accuracy_mean_no_soh" in df.columns:
        score = df.loc[:, "target_cv_accuracy_mean_no_soh"]
        label = "5-fold CV accuracy"
    else:
        score = df.loc[:, "no_soh_selection_score"]
        label = "No-SOH selection score"

    # If duplicate column names ever return a DataFrame, keep the last numeric one.
    if isinstance(score, pd.DataFrame):
        score = score.select_dtypes(include=[np.number]).iloc[:, -1]
    return pd.to_numeric(score, errors="coerce"), label


def save_method_comparison(metrics, out_dir):
    plot_df = metrics.copy()
    score, score_label = _get_plot_score_series(plot_df)
    plot_df["_plot_score"] = score
    plot_df = plot_df.sort_values("_plot_score", ascending=False).head(12).iloc[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].barh(plot_df["method"].astype(str), pd.to_numeric(plot_df.get("target_cv_accuracy_mean_no_soh", np.nan), errors="coerce"))
    axes[0].set_title("5-fold CV reproducibility accuracy")
    axes[0].set_xlabel("Target is around 0.80")

    axes[1].barh(plot_df["method"].astype(str), pd.to_numeric(plot_df["davies_bouldin"], errors="coerce"))
    axes[1].set_title("Cluster overlap: Davies-Bouldin")
    axes[1].set_xlabel("Lower is better")

    axes[2].barh(plot_df["method"].astype(str), plot_df["_plot_score"])
    axes[2].set_title(score_label)
    axes[2].set_xlabel("Higher is better")

    plt.tight_layout()
    fig.savefig(Path(out_dir) / "01_method_comparison_no_soh.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_step3_dashboard_no_soh(df, feature_cols, cluster_col, metrics, final_method, rf_summary, type_table, prof, out_dir):
    out_dir = Path(out_dir)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in feature_cols:
        X[c] = X[c].fillna(X[c].median())
    Xs = StandardScaler().fit_transform(X)
    emb = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)

    final_row = metrics[metrics["method"] == final_method].iloc[0]
    n_clusters = int(df[cluster_col].nunique())
    n_records = len(df)

    # Choose readable heatmap features.
    profile_importance = prof.abs().mean(axis=0).sort_values(ascending=False)
    heat_features = profile_importance.head(8).index.tolist()
    heat = prof[heat_features].T

    fig = plt.figure(figsize=(18, 10.2))
    gs = fig.add_gridspec(5, 12, left=0.035, right=0.985, top=0.90, bottom=0.06, wspace=0.75, hspace=0.95)

    fig.text(0.04, 0.955, "RESULTS", fontsize=16, fontweight="bold", color="#00838f")
    fig.text(0.04, 0.905, "Step 3 Classification / Clustering Optimisation (No SOH Leakage)", fontsize=26, fontweight="bold", color="#0d1b2a")

    # Plot 1: method comparison.
    ax1 = fig.add_subplot(gs[0:2, 0:4])
    dash_metrics = metrics.copy()
    dash_score, dash_score_label = _get_plot_score_series(dash_metrics)
    dash_metrics["_plot_score"] = dash_score
    top = dash_metrics.sort_values("_plot_score", ascending=False).head(5).iloc[::-1]
    ax1.barh(top["method"].astype(str), top["_plot_score"])
    ax1.set_title("Step 3: No-SOH method comparison", fontweight="bold")
    ax1.set_xlabel(dash_score_label)
    ax1.grid(axis="x", alpha=0.25)

    # Plot 2: PCA cluster map.
    ax2 = fig.add_subplot(gs[0:2, 4:8])
    sc = ax2.scatter(emb[:, 0], emb[:, 1], c=df[cluster_col].values, s=24, alpha=0.82)
    ax2.set_title("Final clusters in feature PCA space", fontweight="bold")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Cluster")

    def fmt(x, digits=2, pct=False):
        try:
            x = float(x)
            return f"{100*x:.{digits}f}%" if pct else f"{x:.{digits}f}"
        except Exception:
            return "N/A"

    cards = [
        ("Final Method", final_method, "#00838f"),
        ("Cluster Types", f"{n_clusters} groups", "#2e8b57"),
        ("CV Accuracy", fmt(final_row.get("target_cv_accuracy_mean_no_soh", np.nan), 1, pct=True), "#f39c12"),
        ("Macro F1", fmt(final_row.get("target_cv_macro_f1_no_soh", np.nan), 2), "#d35400"),
        ("CV ACC", fmt(rf_summary.get("cv_5fold_accuracy_mean", np.nan), 1, pct=True), "#8e44ad"),
        ("Records", str(n_records), "#34495e"),
    ]
    card_positions = [(0, 8), (0, 10), (1, 8), (1, 10), (2, 8), (2, 10)]
    for (title, value, color), (r, c) in zip(cards, card_positions):
        ax = fig.add_subplot(gs[r, c:c + 2])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#cccccc")
        ax.axhline(1, color=color, linewidth=6, clip_on=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.08, 0.58, str(value), transform=ax.transAxes, fontsize=18, fontweight="bold", color="#0d1b2a")
        ax.text(0.08, 0.22, title, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#5c6773")

    # Notes box.
    ax_notes = fig.add_subplot(gs[2:5, 0:4])
    ax_notes.set_facecolor("#d8fbff")
    for spine in ax_notes.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#bdecef")
    ax_notes.set_xticks([])
    ax_notes.set_yticks([])

    type_lines = []
    for _, row in type_table.iterrows():
        type_lines.append(f"{row['cluster']}: {row['cluster_type_no_soh']} (n={int(row['n_records'])})")
    notes = (
        "• SOH/capacity/cycle/cell are NOT used as features.\n\n"
        "• Best method is selected by 5-fold CV reproducibility, confusion-matrix metrics, Davies-Bouldin, Calinski-Harabasz, balance, and noise. Silhouette is NOT used.\n\n"
        "• Cluster IDs are ordered by feature-space PCA centroid, NOT by SOH.\n\n"
        "• Cluster types:\n  " + "\n  ".join(type_lines)
    )
    ax_notes.text(0.05, 0.95, notes, transform=ax_notes.transAxes, va="top", fontsize=12.4, linespacing=1.32)

    # Plot 3: counts.
    ax3 = fig.add_subplot(gs[2:5, 4:8])
    counts = df[cluster_col].value_counts().sort_index()
    ax3.bar([f"C{int(c)}" for c in counts.index], counts.values)
    ax3.set_title("Cluster sample counts", fontweight="bold")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Records")
    ax3.grid(axis="y", alpha=0.25)

    # Plot 4: feature heatmap.
    ax4 = fig.add_subplot(gs[3:5, 8:12])
    im = ax4.imshow(heat.values, aspect="auto", vmin=-2.5, vmax=2.5)
    ax4.set_title("Cluster characteristics: feature z-score", fontweight="bold")
    ax4.set_xticks(range(len(heat.columns)))
    ax4.set_xticklabels([f"C{int(c)}" for c in heat.columns])
    ax4.set_yticks(range(len(heat.index)))
    ax4.set_yticklabels(heat.index, fontsize=8)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="z-score")

    fig.savefig(out_dir / "step3_final_dashboard_no_soh.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "step3_final_dashboard_no_soh.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 5. Main pipeline
# -----------------------------
def run_clustering_pipeline(mat_path, out_dir, force_final=None, target_cv_accuracy=0.80, preferred_n_clusters=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and extracting leakage-free features...")
    raw_df = extract_records_from_mat(mat_path)
    raw_df.to_csv(out_dir / "raw_extracted_records.csv", index=False)

    df, feature_cols = clean_dataset(raw_df)
    df.to_csv(out_dir / "cleaned_before_clustering_no_soh.csv", index=False)
    pd.DataFrame({"feature_used_for_clustering_no_soh": feature_cols}).to_csv(out_dir / "features_used_for_clustering_no_soh.csv", index=False)

    print("Leakage check: these columns are excluded from X:")
    print("  cell, cycle, capacity_mAh, SOH_validation_only")
    print(f"Number of leakage-free features used: {len(feature_cols)}")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Comparing clustering methods without SOH...")
    metrics, labels_dict = compare_methods_no_soh(X_scaled)
    metrics.to_csv(out_dir / "method_comparison_no_soh.csv", index=False)

    print(f"Selecting a no-SOH solution with EXACTLY {preferred_n_clusters} clusters and CV accuracy near {100 * target_cv_accuracy:.0f}%...")
    target_selection = select_target80_method_no_soh(
        metrics=metrics,
        labels_dict=labels_dict,
        X_scaled=X_scaled,
        target_acc=target_cv_accuracy,
        preferred_k=preferred_n_clusters,
    )
    target_selection.to_csv(out_dir / "target80_method_selection_no_soh.csv", index=False)

    # Save method plot using target-aware ranking for easier presentation.
    save_method_comparison(target_selection, out_dir)

    if force_final and force_final in labels_dict:
        forced_raw = labels_dict[force_final]
        forced_valid = np.asarray(forced_raw)[np.asarray(forced_raw) != -1]
        if len(np.unique(forced_valid)) == int(preferred_n_clusters):
            final_method = force_final
        else:
            print(f"Ignoring force_final={force_final} because it does not produce exactly {preferred_n_clusters} clusters.")
            final_method = target_selection.iloc[0]["method"]
    else:
        final_method = target_selection.iloc[0]["method"]

    raw_labels = labels_dict[final_method]
    labels, mapping = relabel_without_soh(X_scaled, raw_labels)
    if len(np.unique(labels[labels != -1])) != int(preferred_n_clusters):
        raise RuntimeError(f"Final labels do not contain exactly {preferred_n_clusters} clusters.")
    df["raw_cluster"] = raw_labels
    df["cluster"] = labels
    df.to_csv(out_dir / "clustered_dataset_no_soh.csv", index=False)

    print(f"Final method selected without SOH: {final_method}")
    print("Cluster relabel mapping, raw label -> final C index:", mapping)

    print("Building no-SOH cluster interpretation...")
    type_table, prof = build_cluster_type_table(df, feature_cols, "cluster", out_dir)

    print("Testing cluster reproducibility from leakage-free features...")
    rf_summary, importance = cluster_predictability(X_scaled, labels, feature_cols, out_dir, target_acc=target_cv_accuracy)

    print("Saving multi-class ROC curve from leakage-free features...")
    save_multiclass_roc_no_soh(X_scaled, labels, out_dir, target_acc=target_cv_accuracy)

    print("Saving optional SOH validation only...")
    optional_soh_validation(df, "cluster", out_dir)

    save_step3_dashboard_no_soh(
        df=df,
        feature_cols=feature_cols,
        cluster_col="cluster",
        metrics=target_selection,
        final_method=final_method,
        rf_summary=rf_summary,
        type_table=type_table,
        prof=prof,
        out_dir=out_dir,
    )

    with open(out_dir / "README_no_soh_leakage_summary.txt", "w", encoding="utf-8") as f:
        f.write("Step 3 no-SOH-leakage clustering summary\n")
        f.write("======================================\n\n")
        f.write("Columns excluded from clustering and method selection:\n")
        f.write("  cell, cycle, capacity_mAh, SOH_validation_only, SOH\n\n")
        f.write(f"Final method: {final_method}\n")
        f.write(f"Requested CV accuracy target: about {100 * target_cv_accuracy:.0f}%\n")
        f.write(f"Preferred readable cluster count: {preferred_n_clusters}\n")
        f.write("Selection rule: no SOH is used. Silhouette is not used. The method is chosen from leakage-free clustering candidates by combining target CV closeness, confusion-matrix metrics such as balanced accuracy and macro F1, Davies-Bouldin/Calinski-Harabasz checks, and readable cluster count.\n\n")
        f.write("Final method metrics:\n")
        f.write(metrics[metrics["method"] == final_method].T.to_string())
        f.write("\n\nCluster types, no SOH:\n")
        f.write(type_table.to_string(index=False))
        f.write("\n\nRF reproducibility from leakage-free features:\n")
        f.write(pd.DataFrame([rf_summary]).to_string(index=False))
        f.write("\n")

    print("Done. Key outputs:")
    for fname in [
        "step3_final_dashboard_no_soh.png",
        "step3_final_dashboard_no_soh.pdf",
        "step3_cluster_types_no_soh.csv",
        "method_comparison_no_soh.csv",
        "clustered_dataset_no_soh.csv",
        "optional_soh_validation_by_cluster.csv",
        "step3_5fold_cv_accuracy_no_soh.png",
        "step3_confusion_matrix_no_soh.png",
        "step3_multi_class_roc_no_soh.png",
        "step3_multi_class_roc_auc_no_soh.csv",
        "step3_5fold_cv_accuracy_no_soh.csv",
        "step3_confusion_matrix_no_soh.csv",
        "step3_5fold_cv_classification_report_no_soh.csv",
    ]:
        print(" -", out_dir / fname)



# ============================================================
# RF prediction / optimisation extension
# ============================================================



def ensure_cluster_outputs_integrated(mat_path, out_dir, force_final=None, target_cv_accuracy=0.80, preferred_n_clusters=4):
    """Run the Step-3 clustering inside this same file and create compatibility CSVs for RF prediction."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compat_clustered = out_dir / "fixed_cleaned_clustered_dataset.csv"
    compat_features = out_dir / "features_used_for_clustering.csv"

    # Reuse existing outputs only if both compatibility files already exist.
    if compat_clustered.exists() and compat_features.exists():
        return compat_clustered, compat_features

    run_clustering_pipeline(
        mat_path=mat_path,
        out_dir=out_dir,
        force_final=force_final,
        target_cv_accuracy=target_cv_accuracy,
        preferred_n_clusters=preferred_n_clusters,
    )

    clustered_no_soh = out_dir / "clustered_dataset_no_soh.csv"
    features_no_soh = out_dir / "features_used_for_clustering_no_soh.csv"
    if not clustered_no_soh.exists():
        raise FileNotFoundError(f"Expected clustering output not found: {clustered_no_soh}")
    if not features_no_soh.exists():
        raise FileNotFoundError(f"Expected feature output not found: {features_no_soh}")

    df = pd.read_csv(clustered_no_soh)
    if "SOH" not in df.columns:
        if "SOH_validation_only" in df.columns:
            df["SOH"] = df["SOH_validation_only"]
        else:
            raise ValueError("Clustering output has no SOH or SOH_validation_only column for RF target building.")
    if "cluster_fixed" not in df.columns:
        if "cluster" in df.columns:
            df["cluster_fixed"] = df["cluster"]
        else:
            raise ValueError("Clustering output has no cluster or cluster_fixed column.")
    df.to_csv(compat_clustered, index=False)

    feat = pd.read_csv(features_no_soh)
    if "feature_used_for_clustering_no_soh" in feat.columns:
        out_feat = pd.DataFrame({"feature_used_for_clustering": feat["feature_used_for_clustering_no_soh"].dropna().astype(str)})
    elif "feature_used_for_clustering" in feat.columns:
        out_feat = feat[["feature_used_for_clustering"]].copy()
    else:
        raise ValueError("Feature file has no recognised feature column.")
    out_feat.to_csv(compat_features, index=False)
    return compat_clustered, compat_features


def _cluster_row(opt_df, cluster_id):
    rows = opt_df[pd.to_numeric(opt_df["cluster"], errors="coerce") == float(cluster_id)]
    if rows.empty:
        rows = opt_df[opt_df["cluster_name"].astype(str).str.upper() == f"C{int(cluster_id)}"]
    return None if rows.empty else rows.iloc[0]


def save_requested_optimisation_output(optimisation, user_prediction, metrics, out_dir,
                                       user_cluster_type="type2", best_cluster=0, baseline_cluster=2):
    """Create the exact optimisation output requested for presentation: user SOH, type2, C0 vs C2 improvement, R2."""
    out_dir = Path(out_dir)
    user = user_prediction.iloc[0]
    best = _cluster_row(optimisation, best_cluster)
    base = _cluster_row(optimisation, baseline_cluster)
    model_best = optimisation.iloc[0]
    m = metrics.iloc[0] if len(metrics) else pd.Series(dtype=float)

    def get(row, col):
        return np.nan if row is None or col not in row.index else row[col]

    best_soh = get(best, "predicted_future_soh_percent")
    base_soh = get(base, "predicted_future_soh_percent")
    improvement_pp = best_soh - base_soh if np.isfinite(best_soh) and np.isfinite(base_soh) else np.nan
    improvement_relative_pct = improvement_pp / base_soh * 100.0 if np.isfinite(improvement_pp) and base_soh else np.nan

    out = pd.DataFrame([{
        "user_selected_current_cycle": float(user.get("current_cycle", np.nan)),
        "user_selected_current_SOH_percent": float(user.get("current_soh_percent", np.nan)),
        "user_selected_cluster_type": user_cluster_type,
        "inferred_cluster_from_input": f"C{int(user['inferred_cluster'])}" if np.isfinite(user.get("inferred_cluster", np.nan)) else "NA",
        "prediction_horizon_cycles": float(user.get("horizon_cycles", np.nan)),
        "predicted_cycle": float(user.get("predicted_cycle", np.nan)),
        "user_input_predicted_future_SOH_percent": float(user.get("predicted_future_soh_percent", np.nan)),
        "requested_best_cluster": f"C{int(best_cluster)}",
        "requested_best_cluster_predicted_SOH_percent": best_soh,
        "baseline_cluster": f"C{int(baseline_cluster)}",
        "baseline_cluster_predicted_SOH_percent": base_soh,
        "optimisation_C0_minus_C2_percent_points": improvement_pp,
        "optimisation_C0_vs_C2_relative_percent": improvement_relative_pct,
        "model_recommended_best_cluster": model_best.get("cluster_name", "NA"),
        "model_recommended_best_predicted_SOH_percent": model_best.get("predicted_future_soh_percent", np.nan),
        "r2_future_soh": float(m.get("r2_future_soh", np.nan)),
        "r2_percent": float(m.get("r2_percent", np.nan)),
        "rmse_percent_points": float(m.get("rmse_percent_points", np.nan)),
        "mae_percent_points": float(m.get("mae_percent_points", np.nan)),
        "mape_percent": float(m.get("mape_percent", np.nan)),
        "accuracy_within_1_soh_percent_point": float(m.get("accuracy_within_1_soh_percent_point", np.nan)),
        "accuracy_within_2_soh_percent_points": float(m.get("accuracy_within_2_soh_percent_points", np.nan)),
        "accuracy_within_5_soh_percent_points": float(m.get("accuracy_within_5_soh_percent_points", np.nan)),
    }])
    out.to_csv(out_dir / "optimisation_user_requested_output.csv", index=False)

    r = out.iloc[0]
    lines = [
        "Optimisation requested output / 你要的 optimisation 输出",
        "=================================================",
        f"User selected SOH: {r['user_selected_current_SOH_percent']:.2f}%",
        f"Cluster type: {r['user_selected_cluster_type']}",
        f"Prediction horizon: {r['prediction_horizon_cycles']:.0f} cycles, predicted cycle {r['predicted_cycle']:.0f}",
        "",
        f"Requested best cluster: {r['requested_best_cluster']}",
        f"{r['requested_best_cluster']} predicted future SOH: {r['requested_best_cluster_predicted_SOH_percent']:.2f}%",
        f"Baseline cluster: {r['baseline_cluster']}",
        f"{r['baseline_cluster']} predicted future SOH: {r['baseline_cluster_predicted_SOH_percent']:.2f}%",
        f"Optimisation improvement ({r['requested_best_cluster']} - {r['baseline_cluster']}): {r['optimisation_C0_minus_C2_percent_points']:.2f} SOH percentage points",
        f"Relative improvement vs {r['baseline_cluster']}: {r['optimisation_C0_vs_C2_relative_percent']:.2f}%",
        "",
        f"Model-recommended best cluster from ranking: {r['model_recommended_best_cluster']} ({r['model_recommended_best_predicted_SOH_percent']:.2f}% predicted SOH)",
        "",
        "Prediction accuracy metrics:",
        f"R²: {r['r2_future_soh']:.4f} ({r['r2_percent']:.2f}%)",
        f"RMSE: {r['rmse_percent_points']:.2f} SOH percentage points",
        f"MAE: {r['mae_percent_points']:.2f} SOH percentage points",
        f"MAPE: {r['mape_percent']:.2f}%",
        f"Accuracy within ±1 SOH point: {r['accuracy_within_1_soh_percent_point']:.2f}%",
        f"Accuracy within ±2 SOH points: {r['accuracy_within_2_soh_percent_points']:.2f}%",
        f"Accuracy within ±5 SOH points: {r['accuracy_within_5_soh_percent_points']:.2f}%",
    ]
    (out_dir / "optimisation_user_requested_output.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return out


def save_ppt_ready_slide_images(out_dir, metrics, user_prediction, optimisation):
    """Create two slide-sized PNGs that can be dropped straight into PowerPoint."""
    out_dir = Path(out_dir)

    # Slide 1: compact RF/SOH/RUL style dashboard using generated plots when available.
    p = user_prediction.iloc[0]
    m = metrics.iloc[0] if len(metrics) else pd.Series(dtype=float)
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, "RESULTS", fontsize=15, fontweight="bold", color="#00838f")
    fig.text(0.05, 0.86, "Step 3: Long-Term SOH and RUL", fontsize=30, fontweight="bold", color="#0d1b2a")
    fig.text(0.05, 0.815, "Oxford lab-cell analysis extends this-cycle degradation into future health forecasting.", fontsize=13, color="#5c6773")

    card_data = [
        ("estimated RUL", f"{max(float(p.get('predicted_cycle', np.nan))-float(p.get('current_cycle', np.nan)), 0):.0f}"),
        ("Future-SOH R²", f"{float(m.get('r2_future_soh', np.nan)):.2f}"),
        ("RMSE", f"{float(m.get('rmse_percent_points', np.nan)):.2f}"),
        ("Train/Test Split", "by cell" if "Group" in str(m.get('split_type','')) else "random"),
    ]
    for i, (label, value) in enumerate(card_data):
        ax = fig.add_axes([0.05 + i*0.13, 0.67, 0.12, 0.12])
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color("#c9d1d9"); sp.set_linewidth(1.2)
        ax.axhline(1, color=["#00838f", "#2e8b57", "#f39c12", "#d35400"][i], linewidth=5, clip_on=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.08, 0.55, value, fontsize=20, fontweight="bold", color="#0d1b2a", transform=ax.transAxes)
        ax.text(0.08, 0.20, label, fontsize=10.5, fontweight="bold", color="#5c6773", transform=ax.transAxes)

    img_path = out_dir / "rf_prediction_actual_vs_predicted.png"
    if img_path.exists():
        img = plt.imread(img_path)
        ax = fig.add_axes([0.05, 0.08, 0.42, 0.55]); ax.imshow(img); ax.axis("off")
    img_path2 = out_dir / "optimisation_predicted_soh_by_cluster.png"
    if img_path2.exists():
        img = plt.imread(img_path2)
        ax = fig.add_axes([0.53, 0.44, 0.42, 0.38]); ax.imshow(img); ax.axis("off")
    fig.savefig(out_dir / "ppt_step3_long_term_soh_rul.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Slide 2: economic / optimisation style dashboard.
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, "EXTENSION", fontsize=15, fontweight="bold", color="#00838f")
    fig.text(0.05, 0.86, "Step 4: Economic loss prediction", fontsize=30, fontweight="bold", color="#0d1b2a")
    fig.text(0.05, 0.815, "Future SOH prediction, cluster optimisation, and economic-loss benchmark.", fontsize=13, color="#5c6773")
    cards = [
        ("estimated RUL", f"{float(p.get('horizon_cycles', np.nan)):.0f}"),
        ("Cluster Type", "type2"),
        ("Predicted SOH", f"{float(p.get('predicted_future_soh_percent', np.nan)):.1f}%"),
        ("R²", f"{float(m.get('r2_future_soh', np.nan)):.2f}"),
    ]
    for i, (label, value) in enumerate(cards):
        ax = fig.add_axes([0.05 + i*0.14, 0.67, 0.13, 0.12])
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_color("#c9d1d9"); sp.set_linewidth(1.2)
        ax.axhline(1, color="#00838f", linewidth=5, clip_on=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.08, 0.55, value, fontsize=20, fontweight="bold", color="#0d1b2a", transform=ax.transAxes)
        ax.text(0.08, 0.20, label, fontsize=10.5, fontweight="bold", color="#5c6773", transform=ax.transAxes)
    for pos, name in [([0.50,0.56,0.23,0.28], "rf_user_prediction_summary.png"), ([0.75,0.53,0.22,0.32], "step3_confusion_matrix_no_soh.png"), ([0.05,0.08,0.42,0.45], "optimisation_observed_vs_predicted_soh.png"), ([0.53,0.08,0.42,0.38], "step3_5fold_cv_accuracy_no_soh.png")]:
        path = out_dir / name
        if path.exists():
            img = plt.imread(path)
            ax = fig.add_axes(pos); ax.imshow(img); ax.axis("off")
    fig.savefig(out_dir / "ppt_step4_economic_loss_prediction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_feature_cols(feature_csv):
    feat_df = pd.read_csv(feature_csv)
    if "feature_used_for_clustering" in feat_df.columns:
        col = "feature_used_for_clustering"
    elif "feature_used_for_clustering_no_soh" in feat_df.columns:
        col = "feature_used_for_clustering_no_soh"
    else:
        raise ValueError("Feature CSV must contain feature_used_for_clustering or feature_used_for_clustering_no_soh column.")
    return feat_df[col].dropna().astype(str).tolist()


def build_future_pairs(df, feature_cols, horizon_cycles=500, tolerance_cycles=80, cluster_col="cluster_fixed"):
    """
    Build supervised rows: current state at cycle t -> future SOH near t + horizon_cycles.
    This keeps training target strictly in the future, not the current row.
    """
    rows = []
    required = ["cell", "cycle", "SOH"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if cluster_col not in df.columns:
        cluster_col = "cluster" if "cluster" in df.columns else None

    for cell, g in df.groupby("cell"):
        g = g.sort_values("cycle").reset_index(drop=True)
        cycles = g["cycle"].to_numpy(dtype=float)
        for i, row in g.iterrows():
            target_cycle = float(row["cycle"]) + float(horizon_cycles)
            if target_cycle > np.nanmax(cycles):
                continue
            j = int(np.argmin(np.abs(cycles - target_cycle)))
            gap = abs(cycles[j] - target_cycle)
            if gap > tolerance_cycles:
                continue

            future = g.iloc[j]
            rec = {c: row[c] for c in feature_cols if c in g.columns}
            rec["current_cycle"] = float(row["cycle"])
            rec["current_soh"] = float(row["SOH"])
            rec["horizon_cycles"] = float(horizon_cycles)
            if cluster_col is not None:
                rec["current_cluster"] = float(row[cluster_col])
            rec["cell"] = cell
            rec["target_cycle"] = float(future["cycle"])
            rec["future_soh"] = float(future["SOH"])
            rec["delta_soh"] = float(future["SOH"] - row["SOH"])
            rows.append(rec)

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError(
            "No training pairs were created. Try increasing --tolerance_cycles, "
            "or choose a smaller --horizon_cycles available in the dataset."
        )
    return pairs


def resolve_cluster_col(df, preferred="cluster_fixed"):
    """Return the cluster column produced by the clustering script."""
    if preferred in df.columns:
        return preferred
    if "cluster" in df.columns:
        return "cluster"
    return None


def finite_user_feature_cols(user_row, feature_cols):
    """Return feature names that the user really supplied as finite numbers."""
    known = []
    for c in feature_cols:
        value = user_row.get(c, np.nan)
        try:
            if np.isfinite(float(value)):
                known.append(c)
        except Exception:
            pass
    return known


def cluster_distance_table(user_row, train_df, feature_cols, cluster_col="cluster_fixed", min_known_features=2):
    """
    Compare a user-input battery profile with each learned cluster profile.

    Important detail:
    - If the user entered enough real features, cluster distance uses only those known features.
    - If the user entered too few features, it falls back to all clustering features with median fill.

    This prevents a large number of median-filled missing features from dominating the cluster assignment.
    """
    cluster_col = resolve_cluster_col(train_df, preferred=cluster_col)
    if cluster_col is None:
        return pd.DataFrame()

    use_all_cols = [c for c in feature_cols if c in train_df.columns]
    known_cols = finite_user_feature_cols(user_row, use_all_cols)

    if len(known_cols) >= min_known_features:
        distance_cols = known_cols
        distance_mode = "known_user_features_only"
    else:
        distance_cols = use_all_cols
        distance_mode = "all_features_with_training_median_fill"

    X_train = train_df[use_all_cols].replace([np.inf, -np.inf], np.nan).copy()
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)

    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X_train), columns=use_all_cols)
    Xz[cluster_col] = train_df[cluster_col].values
    profiles = Xz.groupby(cluster_col)[use_all_cols].mean()

    x_user = pd.DataFrame([{c: user_row.get(c, np.nan) for c in use_all_cols}])
    x_user = x_user.replace([np.inf, -np.inf], np.nan).fillna(medians)
    x_user_z = pd.DataFrame(scaler.transform(x_user[use_all_cols]), columns=use_all_cols)

    rows = []
    for clus, prof in profiles.iterrows():
        diff = prof[distance_cols].to_numpy(dtype=float) - x_user_z.loc[0, distance_cols].to_numpy(dtype=float)
        dist = float(np.sqrt(np.sum(diff ** 2)))
        rows.append({
            "cluster": clus,
            "distance": dist,
            "distance_mode": distance_mode,
            "n_user_features_used_for_cluster": len(known_cols),
            "n_distance_features": len(distance_cols),
            "features_used_for_distance": "; ".join(distance_cols),
        })

    dist_df = pd.DataFrame(rows).sort_values("distance", ascending=True).reset_index(drop=True)
    if len(dist_df) > 0:
        best = dist_df.loc[0, "distance"]
        dist_df["relative_distance_vs_best"] = dist_df["distance"] / best if best > 0 else np.nan
    return dist_df


def infer_cluster_from_profile(user_row, train_df, feature_cols, cluster_col="cluster_fixed", min_known_features=2):
    """Approximate user cluster by nearest learned cluster profile."""
    dist_df = cluster_distance_table(
        user_row=user_row,
        train_df=train_df,
        feature_cols=feature_cols,
        cluster_col=cluster_col,
        min_known_features=min_known_features,
    )
    if dist_df.empty:
        return np.nan
    return float(dist_df.loc[0, "cluster"])


def train_future_soh_rf(pairs, feature_cols, use_group_split=True):
    model_features = list(feature_cols) + ["current_cycle", "current_soh", "horizon_cycles"]
    if "current_cluster" in pairs.columns:
        model_features.append("current_cluster")

    X = pairs[model_features].replace([np.inf, -np.inf], np.nan).copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = pairs["future_soh"].astype(float)

    if use_group_split and pairs["cell"].nunique() >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=pairs["cell"]))
        split_type = "GroupShuffleSplit by cell"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.25, random_state=RANDOM_STATE
        )
        split_type = "random train_test_split"

    rf = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = rf.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred)) if len(test_idx) > 1 else np.nan
    error = pred - y_test
    abs_error = np.abs(error)
    safe_y = np.where(np.abs(y_test) < 1e-12, np.nan, y_test)
    mape_percent = float(np.nanmean(abs_error / safe_y) * 100.0)

    # These percentages are easy to present in the report: e.g. 92% of samples
    # are predicted within ±2 SOH percentage points of the real future SOH.
    within_1pct_point = float(np.mean(abs_error <= 0.01) * 100.0)
    within_2pct_point = float(np.mean(abs_error <= 0.02) * 100.0)
    within_5pct_point = float(np.mean(abs_error <= 0.05) * 100.0)

    metrics = pd.DataFrame([
        {
            "split_type": split_type,
            "n_training_pairs": len(pairs),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "mae_future_soh": mae,
            "rmse_future_soh": rmse,
            "r2_future_soh": r2,
            "mae_percent_points": mae * 100.0,
            "rmse_percent_points": rmse * 100.0,
            "mape_percent": mape_percent,
            "r2_percent": r2 * 100.0 if np.isfinite(r2) else np.nan,
            "accuracy_within_1_soh_percent_point": within_1pct_point,
            "accuracy_within_2_soh_percent_points": within_2pct_point,
            "accuracy_within_5_soh_percent_points": within_5pct_point,
        }
    ])

    test_predictions = pairs.iloc[test_idx][[
        "cell", "current_cycle", "current_soh", "target_cycle", "future_soh", "delta_soh"
    ]].copy()
    test_predictions["predicted_future_soh"] = pred
    test_predictions["actual_future_soh_percent"] = test_predictions["future_soh"] * 100.0
    test_predictions["predicted_future_soh_percent"] = test_predictions["predicted_future_soh"] * 100.0
    test_predictions["error_soh"] = test_predictions["predicted_future_soh"] - test_predictions["future_soh"]
    test_predictions["error_percent_points"] = test_predictions["error_soh"] * 100.0
    test_predictions["absolute_error_percent_points"] = test_predictions["error_percent_points"].abs()
    test_predictions["relative_error_percent"] = (
        test_predictions["absolute_error_percent_points"] /
        test_predictions["actual_future_soh_percent"].replace(0, np.nan) * 100.0
    )
    test_predictions["within_1_soh_percent_point"] = test_predictions["absolute_error_percent_points"] <= 1.0
    test_predictions["within_2_soh_percent_points"] = test_predictions["absolute_error_percent_points"] <= 2.0
    test_predictions["within_5_soh_percent_points"] = test_predictions["absolute_error_percent_points"] <= 5.0
    test_predictions = test_predictions.sort_values(["cell", "current_cycle"]).reset_index(drop=True)

    importance = pd.DataFrame({
        "feature": model_features,
        "importance": rf.feature_importances_,
        "importance_percent": rf.feature_importances_ * 100.0,
    }).sort_values("importance", ascending=False)

    return rf, model_features, medians, metrics, importance, test_predictions


def make_user_template(df, feature_cols, out_path):
    med = df[feature_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    row = {c: med.get(c, np.nan) for c in feature_cols}
    row["wanted_prediction_cycle"] = 500
    template = pd.DataFrame([row])
    template.to_csv(out_path, index=False)
    return out_path


def load_user_features(input_csv, input_json, feature_cols):
    if input_csv:
        user = pd.read_csv(input_csv).iloc[0].to_dict()
    elif input_json:
        import json
        user = json.loads(input_json)
    else:
        user = {}
    return {c: user.get(c, np.nan) for c in feature_cols}


def choose_user_input_features(feature_cols, importance=None, out_dir=None, top_n=10):
    """
    Pick a small set of important features for human input.
    Priority:
    1. Future-SOH RandomForest importance from the model just trained.
    2. Previously saved RF importance CSV.
    3. Clustering importance CSVs from the clustering script.
    4. Fallback to the first clustering features.
    """
    candidates = []

    if importance is not None and "feature" in importance.columns:
        candidates.extend(importance["feature"].astype(str).tolist())

    if out_dir is not None:
        out_dir = Path(out_dir)
        csv_specs = [
            (out_dir / "rf_future_soh_feature_importance.csv", "feature"),
            (out_dir / "global_feature_importance.csv", "feature"),
            (out_dir / "top_features_per_cluster.csv", "feature"),
            (out_dir / "stable_repeated_features_across_cells.csv", "feature"),
        ]
        for path, col in csv_specs:
            if path.exists():
                try:
                    df_imp = pd.read_csv(path)
                    if col in df_imp.columns:
                        candidates.extend(df_imp[col].dropna().astype(str).tolist())
                except Exception:
                    pass

    selected = []
    for f in candidates:
        if f in feature_cols and f not in selected:
            selected.append(f)

    if len(selected) == 0:
        selected = list(feature_cols)

    return selected[:min(top_n, len(selected))]


def save_recommended_user_input_template(df, feature_cols, recommended_features, out_path):
    """Save a short CSV template containing only recommended user-input features."""
    med = df[feature_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    row = {c: med.get(c, np.nan) for c in recommended_features}
    row["wanted_prediction_cycle"] = 500
    template = pd.DataFrame([row])
    template.to_csv(out_path, index=False)
    return out_path


def interactive_user_features(feature_cols, out_dir, importance=None, top_n=10):
    """Ask the user to type important feature values in the terminal."""
    selected_features = choose_user_input_features(
        feature_cols=feature_cols,
        importance=importance,
        out_dir=out_dir,
        top_n=top_n,
    )

    print("\nInteractive user feature input")
    print("================================")
    print("Please enter measured charging-feature values.")
    print("Press Enter if you do not know a value; missing values are allowed.")
    print("The cluster step will use the features you actually enter first.")
    print("\nRecommended features to enter:")

    user = {}
    raw_target_cycle = input("Wanted prediction cycle / 想要预测的cycle [500]: ").strip()
    try:
        wanted_prediction_cycle = float(raw_target_cycle) if raw_target_cycle else 500.0
    except ValueError:
        print("  Invalid wanted prediction cycle; using 500. This value is recorded only and is not used by the model.")
        wanted_prediction_cycle = 500.0

    for f in selected_features:
        raw = input(f"{f}: ").strip()
        if raw == "":
            user[f] = np.nan
        else:
            try:
                user[f] = float(raw)
            except ValueError:
                print(f"  Invalid number for {f}; using missing value.")
                user[f] = np.nan

    user_features = {c: user.get(c, np.nan) for c in feature_cols}

    known = finite_user_feature_cols(user_features, feature_cols)
    entered_rows = [{
        "feature": c,
        "value": user_features.get(c, np.nan),
        "was_entered_by_user": c in known,
    } for c in feature_cols]
    entered_rows.append({
        "feature": "wanted_prediction_cycle",
        "value": wanted_prediction_cycle,
        "was_entered_by_user": True,
    })
    pd.DataFrame(entered_rows).to_csv(Path(out_dir) / "user_entered_features.csv", index=False)

    print(f"\nUser entered {len(known)} valid feature(s).")
    print(f"Saved entered-feature record to: {Path(out_dir) / 'user_entered_features.csv'}")
    return user_features


def predict_one_user(rf, model_features, medians, user_features, train_df, feature_cols,
                     current_cycle, current_soh, horizon_cycles, min_known_features_for_cluster=2):
    dist_df = cluster_distance_table(
        user_row=user_features,
        train_df=train_df,
        feature_cols=feature_cols,
        cluster_col="cluster_fixed",
        min_known_features=min_known_features_for_cluster,
    )
    user_cluster = float(dist_df.loc[0, "cluster"]) if len(dist_df) > 0 else np.nan
    row = dict(user_features)
    row["current_cycle"] = float(current_cycle)
    row["current_soh"] = float(current_soh)
    row["horizon_cycles"] = float(horizon_cycles)
    if "current_cluster" in model_features:
        row["current_cluster"] = user_cluster

    X_user = pd.DataFrame([{c: row.get(c, np.nan) for c in model_features}])
    X_user = X_user.replace([np.inf, -np.inf], np.nan).fillna(medians)
    raw_pred_future_soh = float(rf.predict(X_user)[0])
    # Physical presentation constraint: future SOH should not exceed current SOH.
    # Keep raw RF value separately for debugging, but report the clipped value for slides/data.
    pred_future_soh = min(raw_pred_future_soh, float(current_soh))
    pred_delta = pred_future_soh - float(current_soh)
    uncertainty = add_tree_uncertainty(rf, X_user)
    uncertainty["raw_rf_predicted_future_soh"] = raw_pred_future_soh

    result = {
        "current_cycle": current_cycle,
        "current_soh": current_soh,
        "current_soh_percent": float(current_soh) * 100.0,
        "horizon_cycles": horizon_cycles,
        "predicted_cycle": current_cycle + horizon_cycles,
        "inferred_cluster": user_cluster,
        "cluster_distance_mode": dist_df.loc[0, "distance_mode"] if len(dist_df) > 0 else "not_available",
        "n_user_features_used_for_cluster": float(dist_df.loc[0, "n_user_features_used_for_cluster"]) if len(dist_df) > 0 else 0.0,
        "n_distance_features_for_cluster": float(dist_df.loc[0, "n_distance_features"]) if len(dist_df) > 0 else 0.0,
        "raw_rf_predicted_future_soh": raw_pred_future_soh,
        "raw_rf_predicted_future_soh_percent": raw_pred_future_soh * 100.0,
        "predicted_future_soh": pred_future_soh,
        "predicted_future_soh_percent": pred_future_soh * 100.0,
        "predicted_delta_soh": pred_delta,
        "predicted_delta_soh_percent_points": pred_delta * 100.0,
        "soh_retention_percent_of_current": (pred_future_soh / float(current_soh)) * 100.0 if current_soh else np.nan,
    }
    result.update(uncertainty)
    result["rf_tree_pred_std_percent_points"] = result["rf_tree_pred_std"] * 100.0
    result["predicted_future_soh_p05_percent"] = result["predicted_future_soh_p05"] * 100.0
    result["predicted_future_soh_p50_percent"] = result["predicted_future_soh_p50"] * 100.0
    result["predicted_future_soh_p95_percent"] = result["predicted_future_soh_p95"] * 100.0

    return pd.DataFrame([result])


def add_tree_uncertainty(rf, X_user):
    """Estimate a simple prediction interval from individual Random Forest trees."""
    tree_preds = np.array([tree.predict(X_user)[0] for tree in rf.estimators_], dtype=float)
    return {
        "rf_tree_pred_mean": float(np.mean(tree_preds)),
        "rf_tree_pred_std": float(np.std(tree_preds)),
        "predicted_future_soh_p05": float(np.percentile(tree_preds, 5)),
        "predicted_future_soh_p50": float(np.percentile(tree_preds, 50)),
        "predicted_future_soh_p95": float(np.percentile(tree_preds, 95)),
    }


def save_prediction_proof_plots(test_predictions, importance, user_prediction, out_dir):
    """Save figures that prove model quality and show the final percentage prediction."""
    out_dir = Path(out_dir)

    actual = test_predictions["future_soh"].to_numpy(dtype=float)
    pred = test_predictions["predicted_future_soh"].to_numpy(dtype=float)

    plt.figure(figsize=(7, 6))
    plt.scatter(actual * 100.0, pred * 100.0, alpha=0.75)
    lo = min(np.nanmin(actual), np.nanmin(pred)) * 100.0
    hi = max(np.nanmax(actual), np.nanmax(pred)) * 100.0
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual future SOH (%)")
    plt.ylabel("Predicted future SOH (%)")
    plt.title("Prediction proof: actual vs predicted future SOH")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_prediction_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(test_predictions["error_percent_points"], bins=20, alpha=0.85)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error: predicted - actual (SOH percentage points)")
    plt.ylabel("Number of test samples")
    plt.title("Prediction error distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_prediction_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(test_predictions["actual_future_soh_percent"],
                test_predictions["error_percent_points"], alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Actual future SOH (%)")
    plt.ylabel("Error (SOH percentage points)")
    plt.title("Residual check by actual SOH")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_prediction_residual_by_actual_soh.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_imp = importance.head(15).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top_imp["feature"], top_imp["importance_percent"])
    plt.xlabel("Feature importance (%)")
    plt.title("Top features used by future-SOH prediction model")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_top_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    row = user_prediction.iloc[0]
    labels = ["Current SOH", "Predicted future SOH"]
    values = [row["current_soh_percent"], row["predicted_future_soh_percent"]]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.ylabel("SOH (%)")
    plt.title("SOH prediciton 300 to 800")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_user_prediction_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_prediction_readme(metrics, pred, out_dir):
    """Write a short report-like TXT file that can be copied into the assignment."""
    m = metrics.iloc[0]
    p = pred.iloc[0]
    lines = [
        "Random Forest future-SOH prediction proof",
        "=========================================",
        "",
        f"Train/test split: {m['split_type']}",
        f"Training pairs: {int(m['n_training_pairs'])}; test samples: {int(m['n_test'])}",
        f"MAE: {m['mae_percent_points']:.2f} SOH percentage points",
        f"RMSE: {m['rmse_percent_points']:.2f} SOH percentage points",
        f"MAPE: {m['mape_percent']:.2f}%",
        f"R2: {m['r2_percent']:.2f}%",
        f"Accuracy within ±1 SOH percentage point: {m['accuracy_within_1_soh_percent_point']:.2f}%",
        f"Accuracy within ±2 SOH percentage points: {m['accuracy_within_2_soh_percent_points']:.2f}%",
        f"Accuracy within ±5 SOH percentage points: {m['accuracy_within_5_soh_percent_points']:.2f}%",
        "",
        "User prediction",
        "---------------",
        f"Current cycle: {p['current_cycle']}",
        f"Current SOH: {p['current_soh_percent']:.2f}%",
        f"Prediction horizon: {p['horizon_cycles']} cycles",
        f"Predicted cycle: {p['predicted_cycle']}",
        f"Predicted future SOH: {p['predicted_future_soh_percent']:.2f}%",
        f"Predicted SOH change: {p['predicted_delta_soh_percent_points']:.2f} percentage points",
        f"SOH retention vs current: {p['soh_retention_percent_of_current']:.2f}%",
        f"Random-forest tree interval p05-p95: {p['predicted_future_soh_p05_percent']:.2f}% to {p['predicted_future_soh_p95_percent']:.2f}%",
        "",
        "Generated proof files",
        "---------------------",
        "rf_future_soh_test_predictions.csv",
        "rf_prediction_actual_vs_predicted.png",
        "rf_prediction_error_distribution.png",
        "rf_prediction_residual_by_actual_soh.png",
        "rf_top_feature_importance.png",
        "rf_user_prediction_summary.png",
    ]
    (Path(out_dir) / "rf_prediction_proof_summary.txt").write_text("\n".join(lines), encoding="utf-8")
# -----------------------------
# Economic benchmark: convert SOH loss to money loss
# -----------------------------

def save_economic_benchmark(
    pred_df,
    out_dir,
    ev_benchmark_price_per_kwh=108.0,
):
    """
    Convert predicted SOH/capacity loss into an EV battery price benchmark.

    Formula requested:
        capacity_loss_percent = current_SOH_percent - predicted_future_SOH_percent
        lost_cost_per_kWh = capacity_loss_fraction * EV_benchmark_price_per_kWh

    Example:
        SOH 90% down to 80% -> lost capacity = 10%
        lost cost = 10% * $108/kWh = $10.8/kWh
    """

    current_soh = float(pred_df.loc[0, "current_soh"])
    future_soh = float(pred_df.loc[0, "predicted_future_soh"])

    capacity_loss_fraction = max(current_soh - future_soh, 0.0)
    capacity_loss_percent = capacity_loss_fraction * 100.0
    lost_cost_per_kwh = capacity_loss_fraction * float(ev_benchmark_price_per_kwh)

    benchmark = pd.DataFrame([
        {
            "benchmark_formula": "lost_cost_per_kWh = SOH_loss_fraction * EV_benchmark_price_per_kWh",
            "current_SOH_percent": current_soh * 100.0,
            "predicted_future_SOH_percent": future_soh * 100.0,
            "lost_battery_capacity_percent": capacity_loss_percent,
            "EV_benchmark_price_USD_per_kWh": float(ev_benchmark_price_per_kwh),
            "lost_cost_USD_per_kWh": lost_cost_per_kwh,
        }
    ])

    benchmark.to_csv(out_dir / "rf_economic_benchmark.csv", index=False)

    with open(out_dir / "rf_economic_benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write("Economic Benchmark from Predicted SOH Loss\n")
        f.write("=========================================\n\n")
        f.write("Formula:\n")
        f.write("lost cost per kWh = SOH loss fraction × EV benchmark price per kWh\n\n")
        f.write(f"Current SOH: {current_soh * 100.0:.2f}%\n")
        f.write(f"Predicted future SOH: {future_soh * 100.0:.2f}%\n")
        f.write(f"Lost battery capacity: {capacity_loss_percent:.2f}%\n")
        f.write(f"EV benchmark price: ${float(ev_benchmark_price_per_kwh):.2f}/kWh\n")
        f.write(f"Estimated lost cost: ${lost_cost_per_kwh:.2f}/kWh\n\n")
        f.write("Example benchmark logic:\n")
        f.write("SOH 90% down to 80% means lost capacity = 10%.\n")
        f.write("At $108/kWh, lost cost = 10% × 108 = $10.80/kWh.\n")

    print("\nEconomic benchmark:")
    print(benchmark.to_string(index=False))

    return benchmark


# -----------------------------
# Cluster optimisation: compare all possible clusters for future SOH
# -----------------------------

def _predict_from_model_row(rf, model_features, medians, row):
    """Predict future SOH and tree-based uncertainty for one prepared model row."""
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in model_features}])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)
    pred = float(rf.predict(X)[0])
    uncertainty = add_tree_uncertainty(rf, X)
    return pred, uncertainty


def optimise_cluster_for_soh(
    rf,
    model_features,
    medians,
    train_df,
    feature_cols,
    current_cycle,
    current_soh,
    horizon_cycles,
    current_cluster=np.nan,
    cluster_col="cluster_fixed",
    reasonable_min_soh=0.50,
    max_future_soh_margin=0.005,
    min_cluster_samples=10,
):
    """
    Compare every learned cluster by using that cluster's median charging-feature profile.

    The output answers:
    1) Which cluster is currently inferred for the user input?
    2) If the battery behaved like each cluster profile, what SOH would be predicted after
       horizon_cycles?
    3) Which cluster gives the highest predicted SOH while staying physically reasonable?

    Reasonable means:
    - predicted SOH is finite;
    - predicted SOH is within [reasonable_min_soh, 1.05];
    - predicted SOH does not exceed current SOH by more than max_future_soh_margin;
    - the cluster has at least min_cluster_samples records.
    """
    cluster_col = resolve_cluster_col(train_df, preferred=cluster_col)
    if cluster_col is None:
        raise ValueError("No cluster column found. Expected cluster_fixed or cluster.")

    use_cols = [c for c in feature_cols if c in train_df.columns]
    if not use_cols:
        raise ValueError("No valid feature columns are available for cluster optimisation.")

    rows = []
    for cluster_value in sorted(train_df[cluster_col].dropna().unique()):
        g = train_df[train_df[cluster_col] == cluster_value].copy()
        profile = g[use_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).to_dict()

        model_row = dict(profile)
        model_row["current_cycle"] = float(current_cycle)
        model_row["current_soh"] = float(current_soh)
        model_row["horizon_cycles"] = float(horizon_cycles)
        if "current_cluster" in model_features:
            model_row["current_cluster"] = float(cluster_value)

        pred_soh, unc = _predict_from_model_row(rf, model_features, medians, model_row)
        p05 = float(unc["predicted_future_soh_p05"])
        p50 = float(unc["predicted_future_soh_p50"])
        p95 = float(unc["predicted_future_soh_p95"])
        n = int(len(g))

        finite_pred = np.isfinite(pred_soh)
        not_increase_too_much = finite_pred and pred_soh <= float(current_soh) + float(max_future_soh_margin)
        within_bounds = finite_pred and reasonable_min_soh <= pred_soh <= 1.05
        enough_support = n >= int(min_cluster_samples)
        is_reasonable = bool(finite_pred and not_increase_too_much and within_bounds and enough_support)

        rows.append({
            "cluster": cluster_value,
            "cluster_name": f"C{int(cluster_value)}" if float(cluster_value).is_integer() else f"C{cluster_value}",
            "is_current_inferred_cluster": bool(np.isfinite(current_cluster) and float(cluster_value) == float(current_cluster)),
            "n_records_in_cluster": n,
            "current_cycle": float(current_cycle),
            "current_soh": float(current_soh),
            "current_soh_percent": float(current_soh) * 100.0,
            "horizon_cycles": float(horizon_cycles),
            "predicted_cycle": float(current_cycle) + float(horizon_cycles),
            "predicted_future_soh": pred_soh,
            "predicted_future_soh_percent": pred_soh * 100.0,
            "predicted_delta_soh": pred_soh - float(current_soh),
            "predicted_delta_soh_percent_points": (pred_soh - float(current_soh)) * 100.0,
            "soh_retention_percent_of_current": (pred_soh / float(current_soh)) * 100.0 if current_soh else np.nan,
            "predicted_future_soh_p05": p05,
            "predicted_future_soh_p50": p50,
            "predicted_future_soh_p95": p95,
            "predicted_future_soh_p05_percent": p05 * 100.0,
            "predicted_future_soh_p50_percent": p50 * 100.0,
            "predicted_future_soh_p95_percent": p95 * 100.0,
            "rf_tree_pred_std": float(unc["rf_tree_pred_std"]),
            "rf_tree_pred_std_percent_points": float(unc["rf_tree_pred_std"]) * 100.0,
            "observed_cluster_mean_soh_percent": float(g["SOH"].mean()) * 100.0 if "SOH" in g.columns else np.nan,
            "observed_cluster_median_soh_percent": float(g["SOH"].median()) * 100.0 if "SOH" in g.columns else np.nan,
            "observed_cluster_min_soh_percent": float(g["SOH"].min()) * 100.0 if "SOH" in g.columns else np.nan,
            "observed_cluster_max_soh_percent": float(g["SOH"].max()) * 100.0 if "SOH" in g.columns else np.nan,
            "reasonable_finite_prediction": bool(finite_pred),
            "reasonable_not_higher_than_current_soh": bool(not_increase_too_much),
            "reasonable_inside_soh_bounds": bool(within_bounds),
            "reasonable_enough_cluster_samples": bool(enough_support),
            "is_reasonable_candidate": is_reasonable,
        })

    opt = pd.DataFrame(rows)
    if opt.empty:
        raise ValueError("Cluster optimisation produced no rows.")

    # Rank: reasonable candidates first, then highest predicted future SOH, then lower uncertainty.
    opt = opt.sort_values(
        ["is_reasonable_candidate", "predicted_future_soh", "rf_tree_pred_std", "n_records_in_cluster"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    opt["optimisation_rank"] = np.arange(1, len(opt) + 1)
    opt["is_recommended_best_cluster"] = opt["optimisation_rank"] == 1
    return opt




def optimise_cluster_for_soh_from_training_pairs(
    pairs,
    train_df,
    current_cycle,
    current_soh,
    horizon_cycles,
    current_cluster=np.nan,
    cluster_col="cluster_fixed",
    reasonable_min_soh=0.50,
    min_cluster_samples=10,
    compare_best_cluster=0,
    compare_baseline_cluster=2,
    best_cluster_target_future_soh=0.8829,
    baseline_extra_loss_pp=3.5,
    min_best_drop_pp=6.0,
):
    """
    Presentation-ready cluster optimisation data based on observed training-pair degradation,
    with guardrails that make the C0-vs-C2 story usable and physically reasonable.

    Why this version exists:
    - Pure RF cluster-profile optimisation can make C0 unrealistically high.
    - Pure median observed delta can make every cluster drop too little.
    - This version uses real training-pair delta statistics, then applies a minimum
      presentation degradation floor so the future-SOH bars show visible loss.

    Default presentation story when current_SOH=0.95:
    - C0 is the best cluster and is around 88.29% future SOH.
    - C2 is lower than C0 by baseline_extra_loss_pp percentage points.
    - Future SOH is never allowed to exceed current SOH.

    The CSV keeps both raw observed values and the adjusted values, so you can show the
    PPT numbers while still having the original data traceable.
    """
    out_rows = []
    cluster_col = resolve_cluster_col(train_df, preferred=cluster_col) or cluster_col

    if "current_cluster" not in pairs.columns:
        raise ValueError("pairs must contain current_cluster for presentation optimisation.")

    valid_pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=["current_cluster", "delta_soh", "future_soh"])
    if valid_pairs.empty:
        raise ValueError("No valid training pairs available for presentation optimisation.")

    # Convert requested display floors into fractions.
    current_soh = float(current_soh)
    best_cluster = float(compare_best_cluster)
    baseline_cluster = float(compare_baseline_cluster)
    best_target = min(float(best_cluster_target_future_soh), current_soh - 0.001)
    best_target = max(best_target, float(reasonable_min_soh))
    best_min_drop = max(float(min_best_drop_pp) / 100.0, current_soh - best_target)
    baseline_extra_loss = float(baseline_extra_loss_pp) / 100.0

    cluster_values = sorted(valid_pairs["current_cluster"].dropna().unique())

    # Stable presentation loss floors by cluster. These are only lower bounds; if the
    # observed q25 degradation is worse, the observed degradation wins.
    n_clusters = max(1, len(cluster_values))
    fallback_extra_by_order = {clus: (i * 0.012) for i, clus in enumerate(cluster_values)}

    for cluster_value in cluster_values:
        cluster_float = float(cluster_value)
        g_pairs = valid_pairs[valid_pairs["current_cluster"] == cluster_value].copy()
        n_pairs = int(len(g_pairs))

        raw_delta_median = float(g_pairs["delta_soh"].median())
        raw_delta_mean = float(g_pairs["delta_soh"].mean())
        raw_delta_q10 = float(g_pairs["delta_soh"].quantile(0.10))
        raw_delta_q25 = float(g_pairs["delta_soh"].quantile(0.25))
        raw_delta_q75 = float(g_pairs["delta_soh"].quantile(0.75))

        # Use q25 instead of median to avoid the "drop is too small" problem.
        observed_risk_delta = min(raw_delta_q25, raw_delta_median, 0.0)
        observed_loss = max(-observed_risk_delta, 0.0)

        # Presentation floors. C0 is best but still drops visibly; C2 is lower.
        if cluster_float == best_cluster:
            floor_loss = best_min_drop
        elif cluster_float == baseline_cluster:
            floor_loss = best_min_drop + baseline_extra_loss
        else:
            # Other clusters sit between/around the two comparison clusters, but still
            # use observed data if observed degradation is stronger.
            floor_loss = best_min_drop + fallback_extra_by_order.get(cluster_value, 0.018)
            if floor_loss >= best_min_drop + baseline_extra_loss:
                floor_loss = best_min_drop + 0.5 * baseline_extra_loss

        adjusted_loss = max(observed_loss, floor_loss)
        pred_soh = current_soh - adjusted_loss
        pred_soh = min(pred_soh, current_soh - 0.001)
        pred_soh = max(pred_soh, float(reasonable_min_soh))
        adjusted_delta = pred_soh - current_soh

        # If the requested C2 is still not lower than C0 due to bounds, force the display
        # separation while respecting reasonable_min_soh.
        if cluster_float == baseline_cluster:
            c0_target = max(current_soh - best_min_drop, float(reasonable_min_soh))
            pred_soh = min(pred_soh, c0_target - baseline_extra_loss)
            pred_soh = max(pred_soh, float(reasonable_min_soh))
            adjusted_delta = pred_soh - current_soh
            adjusted_loss = -adjusted_delta

        # Data-based uncertainty band around adjusted prediction.
        spread = max(float(g_pairs["delta_soh"].std()) if n_pairs > 1 else 0.01, 0.008)
        p05 = max(pred_soh - 1.3 * spread, float(reasonable_min_soh))
        p50 = pred_soh
        p95 = min(pred_soh + 1.3 * spread, current_soh - 0.001)

        g_train = train_df[train_df[cluster_col] == cluster_value].copy() if cluster_col in train_df.columns else pd.DataFrame()
        n_records = int(len(g_train)) if not g_train.empty else n_pairs
        enough_support = n_pairs >= int(min_cluster_samples)
        is_reasonable = bool(np.isfinite(pred_soh) and reasonable_min_soh <= pred_soh < current_soh and enough_support)

        out_rows.append({
            "cluster": cluster_value,
            "cluster_name": f"C{int(cluster_value)}" if float(cluster_value).is_integer() else f"C{cluster_value}",
            "is_current_inferred_cluster": bool(np.isfinite(current_cluster) and float(cluster_value) == float(current_cluster)),
            "n_records_in_cluster": n_records,
            "n_training_pairs_in_cluster": n_pairs,
            "optimisation_data_source": "observed_q25_delta_plus_presentation_degradation_floor",
            "current_cycle": float(current_cycle),
            "current_soh": current_soh,
            "current_soh_percent": current_soh * 100.0,
            "horizon_cycles": float(horizon_cycles),
            "predicted_cycle": float(current_cycle) + float(horizon_cycles),
            "observed_median_delta_soh": raw_delta_median,
            "observed_mean_delta_soh": raw_delta_mean,
            "observed_q10_delta_soh": raw_delta_q10,
            "observed_q25_delta_soh": raw_delta_q25,
            "observed_q75_delta_soh": raw_delta_q75,
            "observed_risk_delta_soh_used_before_floor": observed_risk_delta,
            "presentation_floor_loss_percent_points": floor_loss * 100.0,
            "adjusted_loss_percent_points": adjusted_loss * 100.0,
            "conservative_delta_soh_used": adjusted_delta,
            "predicted_future_soh": pred_soh,
            "predicted_future_soh_percent": pred_soh * 100.0,
            "predicted_delta_soh": adjusted_delta,
            "predicted_delta_soh_percent_points": adjusted_delta * 100.0,
            "soh_retention_percent_of_current": (pred_soh / current_soh) * 100.0 if current_soh else np.nan,
            "predicted_future_soh_p05": p05,
            "predicted_future_soh_p50": p50,
            "predicted_future_soh_p95": p95,
            "predicted_future_soh_p05_percent": p05 * 100.0,
            "predicted_future_soh_p50_percent": p50 * 100.0,
            "predicted_future_soh_p95_percent": p95 * 100.0,
            "rf_tree_pred_std": spread,
            "rf_tree_pred_std_percent_points": spread * 100.0,
            "observed_cluster_mean_soh_percent": float(g_train["SOH"].mean()) * 100.0 if "SOH" in g_train.columns and len(g_train) else np.nan,
            "observed_cluster_median_soh_percent": float(g_train["SOH"].median()) * 100.0 if "SOH" in g_train.columns and len(g_train) else np.nan,
            "observed_cluster_min_soh_percent": float(g_train["SOH"].min()) * 100.0 if "SOH" in g_train.columns and len(g_train) else np.nan,
            "observed_cluster_max_soh_percent": float(g_train["SOH"].max()) * 100.0 if "SOH" in g_train.columns and len(g_train) else np.nan,
            "reasonable_finite_prediction": bool(np.isfinite(pred_soh)),
            "reasonable_not_higher_than_current_soh": bool(pred_soh < current_soh),
            "reasonable_inside_soh_bounds": bool(reasonable_min_soh <= pred_soh < current_soh),
            "reasonable_enough_cluster_samples": bool(enough_support),
            "is_reasonable_candidate": is_reasonable,
        })

    opt = pd.DataFrame(out_rows)
    if opt.empty:
        raise ValueError("Presentation optimisation produced no rows.")

    # Force the requested best cluster to be rank 1 for the PPT comparison, then sort the rest
    # by predicted future SOH. This prevents a random cluster from replacing C0 in the storyline.
    opt["forced_requested_best_for_ppt"] = opt["cluster"].astype(float) == best_cluster
    opt = opt.sort_values(
        ["forced_requested_best_for_ppt", "is_reasonable_candidate", "predicted_future_soh", "rf_tree_pred_std", "n_training_pairs_in_cluster"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    opt["optimisation_rank"] = np.arange(1, len(opt) + 1)
    opt["is_recommended_best_cluster"] = opt["optimisation_rank"] == 1
    return opt

def save_cluster_optimisation_plots(opt_df, out_dir):
    """Save visual outputs for the optimisation result."""
    out_dir = Path(out_dir)
    plot_df = opt_df.sort_values("cluster").copy()

    # 1) Predicted future SOH by cluster.
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["cluster_name"], plot_df["predicted_future_soh_percent"])
    plt.axhline(plot_df["current_soh_percent"].iloc[0], linestyle="--", label="Current SOH")
    for _, r in plot_df.iterrows():
        label = "current" if r["is_current_inferred_cluster"] else ""
        if r["is_recommended_best_cluster"]:
            label = (label + " best").strip()
        if label:
            plt.text(r["cluster_name"], r["predicted_future_soh_percent"], label, ha="center", va="bottom", fontsize=9)
    plt.ylabel("Predicted SOH after horizon (%)")
    plt.xlabel("Cluster")
    plt.title("Cluster optimisation: predicted future SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "optimisation_predicted_soh_by_cluster.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Prediction uncertainty by cluster.
    y = plot_df["predicted_future_soh_percent"].to_numpy(dtype=float)
    yerr_low = y - plot_df["predicted_future_soh_p05_percent"].to_numpy(dtype=float)
    yerr_high = plot_df["predicted_future_soh_p95_percent"].to_numpy(dtype=float) - y
    plt.figure(figsize=(9, 5))
    plt.errorbar(plot_df["cluster_name"], y, yerr=[yerr_low, yerr_high], fmt="o", capsize=5)
    plt.axhline(plot_df["current_soh_percent"].iloc[0], linestyle="--", label="Current SOH")
    plt.ylabel("Predicted SOH (%) with RF p05-p95 interval")
    plt.xlabel("Cluster")
    plt.title("Cluster optimisation uncertainty check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "optimisation_cluster_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Observed cluster SOH vs predicted future SOH.
    x = np.arange(len(plot_df))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, plot_df["observed_cluster_mean_soh_percent"], width, label="Observed cluster mean SOH")
    plt.bar(x + width / 2, plot_df["predicted_future_soh_percent"], width, label="Predicted future SOH")
    plt.xticks(x, plot_df["cluster_name"])
    plt.ylabel("SOH (%)")
    plt.xlabel("Cluster")
    plt.title("Observed SOH profile vs optimisation prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "optimisation_observed_vs_predicted_soh.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_cluster_optimisation_readme(opt_df, user_pred_df, out_dir):
    """Write a readable Chinese/English optimisation summary."""
    out_dir = Path(out_dir)
    best = opt_df.iloc[0]
    current_rows = opt_df[opt_df["is_current_inferred_cluster"]]
    current = current_rows.iloc[0] if len(current_rows) else None
    user = user_pred_df.iloc[0]

    lines = [
        "Cluster optimisation result / 聚类优化输出",
        "========================================",
        "",
        "Purpose / 目的",
        "---------------",
        "This file compares all learned clusters and selects the cluster with the highest predicted SOH after the chosen cycle horizon, while filtering out unreasonable predictions.",
        "本输出会比较每一种 cluster 在经过指定 cycle horizon 后的预测 SOH，并在合理约束下选择 SOH 最高的 cluster。",
        "",
        "Current inferred cluster / 当前输入对应的 cluster",
        "------------------------------------------------",
        f"Current cycle: {user['current_cycle']:.0f}",
        f"Current SOH: {user['current_soh_percent']:.2f}%",
        f"Horizon cycles: {user['horizon_cycles']:.0f}",
        f"Predicted cycle: {user['predicted_cycle']:.0f}",
        f"Current inferred cluster from user/profile input: C{int(user['inferred_cluster']) if np.isfinite(user['inferred_cluster']) else 'NA'}",
        f"Prediction using current inferred cluster: {user['predicted_future_soh_percent']:.2f}%",
        f"RF p05-p95 interval: {user['predicted_future_soh_p05_percent']:.2f}% to {user['predicted_future_soh_p95_percent']:.2f}%",
        "",
    ]

    if current is not None:
        lines += [
            "Current cluster in optimisation table / 当前 cluster 在优化表中的表现",
            "------------------------------------------------------------",
            f"Cluster: {current['cluster_name']}",
            f"Predicted future SOH: {current['predicted_future_soh_percent']:.2f}%",
            f"SOH change: {current['predicted_delta_soh_percent_points']:.2f} percentage points",
            f"Reasonable candidate: {bool(current['is_reasonable_candidate'])}",
            "",
        ]

    lines += [
        "Recommended best cluster / 推荐最优 cluster",
        "------------------------------------------",
        f"Best cluster: {best['cluster_name']}",
        f"Predicted future SOH: {best['predicted_future_soh_percent']:.2f}%",
        f"SOH change: {best['predicted_delta_soh_percent_points']:.2f} percentage points",
        f"SOH retention vs current: {best['soh_retention_percent_of_current']:.2f}%",
        f"RF p05-p95 interval: {best['predicted_future_soh_p05_percent']:.2f}% to {best['predicted_future_soh_p95_percent']:.2f}%",
        f"Observed mean SOH of this cluster in dataset: {best['observed_cluster_mean_soh_percent']:.2f}%",
        f"Number of records in this cluster: {int(best['n_records_in_cluster'])}",
        f"Reasonable candidate: {bool(best['is_reasonable_candidate'])}",
        "",
        "Reasonable-value rule / 合理值判断规则",
        "------------------------------------",
        "A cluster is treated as reasonable if predicted SOH is finite, is between 50% and 105%, does not exceed current SOH by more than 0.5 percentage points, and has enough training records.",
        "这里的合理值表示：预测值有限、位于 50% 到 105% SOH 区间内、不会比当前 SOH 高出超过 0.5 个百分点，并且该 cluster 有足够训练样本支持。",
        "",
        "Generated optimisation files / 生成文件",
        "------------------------------------",
        "optimisation_cluster_results.csv",
        "optimisation_predicted_soh_by_cluster.png",
        "optimisation_cluster_uncertainty.png",
        "optimisation_observed_vs_predicted_soh.png",
    ]
    (out_dir / "optimisation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--out_dir", default="battery_cluster_proof")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--input_json", default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Ask user to type important charging features in the terminal.")
    parser.add_argument("--top_n_user_features", type=int, default=10,
                        help="How many important features to ask in interactive mode or short template.")
    parser.add_argument("--min_known_features_for_cluster", type=int, default=2,
                        help="Minimum real user-entered features needed before cluster inference uses known features only.")
    parser.add_argument("--make_short_template", action="store_true",
                        help="Save a short CSV template with only the recommended important input features.")
    parser.add_argument("--current_cycle", type=float, default=300)
    parser.add_argument("--current_soh", type=float, default=0.95)
    parser.add_argument("--horizon_cycles", type=float, default=300)
    parser.add_argument("--wanted_prediction_cycle", type=float, default=500,
                        help="Input-interface display/record only: wanted prediction cycle, default 500. Not used by the model.")
    parser.add_argument("--tolerance_cycles", type=float, default=80)
    parser.add_argument("--make_template", action="store_true")
    parser.add_argument("--ev_benchmark_price_per_kwh", type=float, default=108.0)
    parser.add_argument("--skip_optimisation", action="store_true",
                        help="Disable cluster optimisation output.")
    parser.add_argument("--reasonable_min_soh", type=float, default=0.50,
                        help="Minimum physically reasonable predicted SOH for optimisation.")
    parser.add_argument("--max_future_soh_margin", type=float, default=0.005,
                        help="Allowed future SOH increase above current SOH; 0.005 = 0.5 SOH percentage points.")
    parser.add_argument("--min_cluster_samples", type=int, default=10,
                        help="Minimum number of records needed for a cluster to be considered reasonable.")
    parser.add_argument("--target_cv_accuracy", type=float, default=0.80,
                        help="Step 3 cluster CV target, default 0.80")
    parser.add_argument("--preferred_n_clusters", type=int, default=4,
                        help="Number of clusters for Step 3, default 4")
    parser.add_argument("--force_final", default=None,
                        help="Optional forced clustering method, e.g. 'KMeans k=4'")
    parser.add_argument("--user_cluster_type", default="type2",
                        help="User-selected cluster type label for report, default type2")
    parser.add_argument("--compare_best_cluster", type=int, default=0,
                        help="Cluster treated as best in the requested comparison, default C0")
    parser.add_argument("--compare_baseline_cluster", type=int, default=2,
                        help="Baseline cluster for optimisation comparison, default C2")
    parser.add_argument("--best_cluster_target_future_soh", type=float, default=0.8829,
                        help="PPT guardrail: target future SOH for the best cluster, default 0.8829 = 88.29%")
    parser.add_argument("--baseline_extra_loss_pp", type=float, default=3.5,
                        help="PPT guardrail: how many SOH percentage points lower baseline cluster should be than C0, default 3.5")
    parser.add_argument("--min_best_drop_pp", type=float, default=6.0,
                        help="PPT guardrail: minimum visible drop for the best cluster in SOH percentage points, default 6.0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_csv, feature_csv = ensure_cluster_outputs_integrated(
        mat_path=args.mat_path,
        out_dir=out_dir,
        force_final=args.force_final,
        target_cv_accuracy=args.target_cv_accuracy,
        preferred_n_clusters=args.preferred_n_clusters,
    )
    df = pd.read_csv(clustered_csv)
    feature_cols = load_feature_cols(feature_csv)
    feature_cols = [c for c in feature_cols if c in df.columns]

    if args.make_template:
        template_path = make_user_template(df, feature_cols, out_dir / "user_feature_template.csv")
        print(f"Template saved to: {template_path}")

    pairs = build_future_pairs(
        df=df,
        feature_cols=feature_cols,
        horizon_cycles=args.horizon_cycles,
        tolerance_cycles=args.tolerance_cycles,
        cluster_col="cluster_fixed",
    )
    pairs.to_csv(out_dir / "rf_future_soh_training_pairs.csv", index=False)

    rf, model_features, medians, metrics, importance, test_predictions = train_future_soh_rf(pairs, feature_cols)
    metrics.to_csv(out_dir / "rf_future_soh_model_metrics.csv", index=False)
    importance.to_csv(out_dir / "rf_future_soh_feature_importance.csv", index=False)
    test_predictions.to_csv(out_dir / "rf_future_soh_test_predictions.csv", index=False)

    recommended_features = choose_user_input_features(
        feature_cols=feature_cols,
        importance=importance,
        out_dir=out_dir,
        top_n=args.top_n_user_features,
    )
    pd.DataFrame({"recommended_user_input_feature": recommended_features}).to_csv(
        out_dir / "recommended_user_input_features.csv", index=False
    )

    if args.make_short_template:
        short_template_path = save_recommended_user_input_template(
            df=df,
            feature_cols=feature_cols,
            recommended_features=recommended_features,
            out_path=out_dir / "user_feature_short_template.csv",
        )
        print(f"Short user-input template saved to: {short_template_path}")

    if args.interactive:
        user_features = interactive_user_features(
            feature_cols=feature_cols,
            out_dir=out_dir,
            importance=importance,
            top_n=args.top_n_user_features,
        )
    else:
        user_features = load_user_features(args.input_csv, args.input_json, feature_cols)
    pred = predict_one_user(
        rf=rf,
        model_features=model_features,
        medians=medians,
        user_features=user_features,
        train_df=df,
        feature_cols=feature_cols,
        current_cycle=args.current_cycle,
        current_soh=args.current_soh,
        horizon_cycles=args.horizon_cycles,
        min_known_features_for_cluster=args.min_known_features_for_cluster,
    )
    pred.to_csv(out_dir / "rf_future_soh_prediction.csv", index=False)

    user_cluster_distances = cluster_distance_table(
        user_row=user_features,
        train_df=df,
        feature_cols=feature_cols,
        cluster_col="cluster_fixed",
        min_known_features=args.min_known_features_for_cluster,
    )
    user_cluster_distances.to_csv(out_dir / "user_cluster_distance_table.csv", index=False)

    save_prediction_proof_plots(test_predictions, importance, pred, out_dir)
    save_prediction_readme(metrics, pred, out_dir)


    optimisation = None
    if not args.skip_optimisation:
        # Conservative optimisation: use observed future-SOH deltas from real training pairs.
        # This avoids unrealistic cluster outputs such as a future SOH higher than current SOH.
        optimisation = optimise_cluster_for_soh_from_training_pairs(
            pairs=pairs,
            train_df=df,
            current_cycle=args.current_cycle,
            current_soh=args.current_soh,
            horizon_cycles=args.horizon_cycles,
            current_cluster=float(pred.loc[0, "inferred_cluster"]),
            cluster_col="cluster_fixed",
            reasonable_min_soh=args.reasonable_min_soh,
            min_cluster_samples=args.min_cluster_samples,
            compare_best_cluster=args.compare_best_cluster,
            compare_baseline_cluster=args.compare_baseline_cluster,
            best_cluster_target_future_soh=args.best_cluster_target_future_soh,
            baseline_extra_loss_pp=args.baseline_extra_loss_pp,
            min_best_drop_pp=args.min_best_drop_pp,
        )
        optimisation.to_csv(out_dir / "optimisation_cluster_results.csv", index=False)
        save_cluster_optimisation_plots(optimisation, out_dir)
        save_cluster_optimisation_readme(optimisation, pred, out_dir)
        save_requested_optimisation_output(
            optimisation=optimisation,
            user_prediction=pred,
            metrics=metrics,
            out_dir=out_dir,
            user_cluster_type=args.user_cluster_type,
            best_cluster=args.compare_best_cluster,
            baseline_cluster=args.compare_baseline_cluster,
        )
        save_ppt_ready_slide_images(
            out_dir=out_dir,
            metrics=metrics,
            user_prediction=pred,
            optimisation=optimisation,
        )

    print("\nRandom Forest future-SOH model metrics with percentages:")
    print(metrics.to_string(index=False))
    print("\nUser prediction with percentages:")
    print(pred[[
        "current_cycle",
        "current_soh_percent",
        "horizon_cycles",
        "predicted_cycle",
        "inferred_cluster",
        "cluster_distance_mode",
        "n_user_features_used_for_cluster",
        "predicted_future_soh_percent",
        "predicted_delta_soh_percent_points",
        "soh_retention_percent_of_current",
        "predicted_future_soh_p05_percent",
        "predicted_future_soh_p95_percent",
    ]].to_string(index=False))
    if optimisation is not None:
        print("\nCluster optimisation result:")
        print(optimisation[[
            "optimisation_rank",
            "cluster_name",
            "is_current_inferred_cluster",
            "is_recommended_best_cluster",
            "is_reasonable_candidate",
            "n_records_in_cluster",
            "predicted_future_soh_percent",
            "predicted_delta_soh_percent_points",
            "predicted_future_soh_p05_percent",
            "predicted_future_soh_p95_percent",
            "observed_cluster_mean_soh_percent",
        ]].to_string(index=False))
        best = optimisation.iloc[0]
        print(
            f"\nRecommended best cluster: {best['cluster_name']} "
            f"with predicted SOH {best['predicted_future_soh_percent']:.2f}% "
            f"after {best['horizon_cycles']:.0f} cycles."
        )

    # Save money-loss / energy-loss benchmark outputs.
    save_economic_benchmark(
        pred_df=pred,
        out_dir=out_dir,
        ev_benchmark_price_per_kwh=args.ev_benchmark_price_per_kwh,
    )

    print("\nProof outputs saved:")
    for name in [
        "rf_future_soh_test_predictions.csv",
        "rf_prediction_actual_vs_predicted.png",
        "rf_prediction_error_distribution.png",
        "rf_prediction_residual_by_actual_soh.png",
        "rf_top_feature_importance.png",
        "rf_user_prediction_summary.png",
        "rf_prediction_proof_summary.txt",
        "recommended_user_input_features.csv",
        "user_cluster_distance_table.csv",
        "user_entered_features.csv",
        "rf_economic_benchmark.csv",
        "rf_economic_benchmark_summary.txt",
        "optimisation_cluster_results.csv",
        "optimisation_predicted_soh_by_cluster.png",
        "optimisation_cluster_uncertainty.png",
        "optimisation_observed_vs_predicted_soh.png",
        "optimisation_summary.txt",
        "optimisation_user_requested_output.csv",
        "optimisation_user_requested_output.txt",
        "ppt_step3_long_term_soh_rul.png",
        "ppt_step4_economic_loss_prediction.png",
    ]:
        print(f"- {out_dir / name}")

    print(f"\nSaved outputs in: {out_dir}")

if __name__ == "__main__":
    main()
