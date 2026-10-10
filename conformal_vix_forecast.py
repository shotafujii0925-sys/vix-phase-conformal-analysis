# -*- coding: utf-8 -*-
"""
時系列コンフォーマル予測: VIXの将来値に対する予測区間を構築し、多角的に評価する。

評価軸:
  1. 目標カバレッジと実現カバレッジ
  2. 平均予測区間幅
  3. 時点ごとの区間幅（時系列プロット）
  4. 相場局面別（VIXフェーズ別）のカバレッジ
  5. 急変前後のカバレッジ（Panicエピソード前後21営業日）
  6. 点予測モデル別の比較（Persistence / Linear / Gradient Boosting）
  7〜8. 4手法の比較（後述）
  9. 学習・較正・評価期間の完全な時系列分割（重複なし、先読みなし）
 10. 手法差の不確実性（ブロックブートストラップ）

比較する4手法（点予測モデル・期間・情報集合・下限クリップ・評価指標を全て統一）:
  A. ACI            : ローリング残差窓 + 適応alpha（Gibbs & Candes 2021の考え方）
  B. RollingFixed   : ローリング残差窓 + 固定alpha  ← Aとの差が「alpha適応の効果」
  C. FixedSplit     : 較正期間で1度だけ算出した分位点を固定（以後21年間更新しない）
                      ← Bとの差が「残差窓を更新する効果」
  D. NaiveBaseline  : モデルを使わない無条件区間（persistence残差の較正期間分位点、固定）

  A vs B : alphaを適応させる効果
  B vs C : 最近の残差で分位点を更新する効果
  A vs C : 両方を組み合わせた効果
  * C は「21年間再較正しない」極端な設定であり、非適応手法一般の代表ではない点に注意。

「区間が広ければカバレッジが高いのは当然」という問題に対応するため、Winkler(interval) score
（幅 + 外れ幅に比例した罰則、小さいほど良い）を全ての切り口で併記し、さらに手法差の
95%信頼区間をブロックブートストラップで推定する。
"""
from __future__ import annotations

import json
from collections import deque

import matplotlib
matplotlib.use("Agg")  # 非対話バックエンドに固定（スクリプト実行で完結させるため）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from data_pipeline import (  # noqa: E402
    DATA_DIR, FIGURES_DIR, VIX_PHASE_LABELS, build_dataset, find_vix_episodes,
)

sns.set_theme(style="whitegrid")
RESULTS: dict = {}

ALPHA_TARGET = 0.10          # 目標カバレッジ 90%（周辺カバレッジ。条件付きカバレッジは保証しない）
GAMMA = 0.01                 # ACIの学習率（Gibbs & Candes推奨レンジ内の値を固定。感度分析は未実施）
ALPHA_MIN, ALPHA_MAX = 0.001, 0.5

# --- 期間分割（学習・較正・評価の完全な時系列分割）---
BURN_IN_DAYS = 2520     # Train: 最初の約10年（モデルの初回学習専用、以後walk-forwardで拡張学習）
CALIBRATION_DAYS = 500  # Calibration: Burn-in直後の約2年（固定分位点の算出・ローリング履歴の初期化専用）
REFIT_EVERY = 252       # walk-forward再学習の間隔（約1年）。計算コストとの妥協で固定、感度分析は未実施
CAL_WINDOW = 500        # ローリング残差窓の長さ（営業日）。固定値、感度分析は未実施

# --- ブロックブートストラップ ---
N_BOOT = 10000
BOOT_SEED = 42
COMPARISON_FAMILY_SIZE = 32  # 2 horizons * 2 models * 4 pairs * 2 metrics

METHODS = ["aci", "rolling_fixed", "fixed_split", "naive"]
METHOD_LABELS = {
    "aci": "A. ACI (rolling window + adaptive alpha)",
    "rolling_fixed": "B. RollingFixed (rolling window + fixed alpha)",
    "fixed_split": "C. FixedSplit (frozen calibration quantile)",
    "naive": "D. NaiveBaseline (model-free, frozen)",
}

FEATURE_COLS = [
    "VIX", "VIX_lag1", "VIX_lag5", "VIX_lag10", "VIX_lag21", "VIX_lag63",
    "VIX_roll_mean_21", "VIX_roll_std_21", "VIX_roll_mean_63", "VIX_roll_std_63",
    "SP500_ret_1", "SP500_realized_vol_21", "Treasury_10Y", "Treasury_10Y_Diff1",
]


# =====================================================================
# 特徴量
# =====================================================================
def build_features(df_raw: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    時刻tまでの情報のみを使った特徴量と、t+h時点のVIXを目的変数として構築する。
    特徴量は全て shift(+k) / rolling()（過去方向のみ）。負のshiftは目的変数にのみ使用する。

    ラグ 1/5/10/21/63 は「前日・1週・2週・1ヶ月・四半期」に対応する慣行的な区切りとして
    探索的に採用したもので、特徴量選択の最適化は行っていない。
    """
    feat = pd.DataFrame(index=df_raw.index)
    vix = df_raw["VIX"]
    sp500_ret = np.log(df_raw["SP500"]).diff()

    feat["VIX"] = vix
    feat["VIX_Phase"] = df_raw["VIX_Phase"]
    for lag in [1, 5, 10, 21, 63]:
        feat[f"VIX_lag{lag}"] = vix.shift(lag)
    feat["VIX_roll_mean_21"] = vix.rolling(21).mean()
    feat["VIX_roll_std_21"] = vix.rolling(21).std()
    feat["VIX_roll_mean_63"] = vix.rolling(63).mean()
    feat["VIX_roll_std_63"] = vix.rolling(63).std()
    feat["SP500_ret_1"] = sp500_ret
    feat["SP500_realized_vol_21"] = sp500_ret.rolling(21).std() * np.sqrt(252)
    feat["Treasury_10Y"] = df_raw["Treasury_10Y"]
    feat["Treasury_10Y_Diff1"] = df_raw["Treasury_10Y_Diff1"]

    feat["target"] = vix.shift(-h)  # 未来値はここでのみ使用
    feat["target_date"] = pd.Series(df_raw.index, index=df_raw.index).shift(-h)
    feat = feat.dropna(subset=[c for c in feat.columns if c != "VIX_Phase"])
    return feat


# =====================================================================
# 点予測（ウォークフォワード）
# =====================================================================
def walk_forward_forecast(feat: pd.DataFrame, model_name: str, h: int) -> pd.Series:
    """
    先読みなしのウォークフォワード点予測。

    学習に使うのは target_date が再学習日以前の行に限る。
    Treasury欠損除去後の行数を営業日数と取り違えない。

    model_name: 'persistence' | 'linear' | 'gbm'
      - persistence: 学習不要のランダムウォーク（h日後もVIXは今日と同じ）
      - linear     : 線形回帰
      - gbm        : HistGradientBoostingRegressor（非線形性を捉える比較対象）
                     max_depth=4, max_iter=200 は計算時間を抑えるために事前に固定した値であり、
                     ハイパーパラメータ探索は一切行っていない（評価期間も一切参照していない）。
    """
    n = len(feat)
    preds = pd.Series(index=feat.index, dtype=float)

    if model_name == "persistence":
        preds[:] = feat["VIX"].values
        return preds

    start = BURN_IN_DAYS
    while start < n:
        train = feat.iloc[:start]
        train = train.loc[train["target_date"] <= feat.index[start]]
        if len(train) < 100:
            raise ValueError("学習データが不足しています")
        end = min(start + REFIT_EVERY, n)
        test_idx = feat.index[start:end]

        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "gbm":
            model = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=42)
        else:
            raise ValueError(model_name)

        model.fit(train[FEATURE_COLS], train["target"])
        preds.loc[test_idx] = model.predict(feat.loc[test_idx, FEATURE_COLS])
        start = end

    return preds


# =====================================================================
# 評価指標
# =====================================================================
def winkler_score(y, lower, upper, alpha=ALPHA_TARGET):
    """
    Winkler(interval) score。
      幅 + (下に外れた分) * 2/alpha + (上に外れた分) * 2/alpha
    値が小さいほど「狭くて、かつ良く当たる」区間。
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = np.clip(lower - y, 0, None) * (2 / alpha)
    above = np.clip(y - upper, 0, None) * (2 / alpha)
    return width + below + above


def evaluate_interval(y, lower, upper, alpha=ALPHA_TARGET, label="") -> dict:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    ws = winkler_score(y, lower, upper, alpha)
    return {
        "label": label,
        "n": int(len(y)),
        "n_covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "winkler_score": float(ws.mean()),
    }


# =====================================================================
# 区間構築（4手法）
# =====================================================================
def conformal_quantile(scores, alpha, *, cap_at_max=False):
    """Finite-sample order statistic: ceil((n+1)*(1-alpha)), one-based.

    FixedSplit returns infinity when the required rank exceeds n. The adaptive
    rolling variant explicitly caps at the largest residual for finite output;
    that practical cap does not carry a distribution-free coverage guarantee.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite, nonempty calibration scores are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    if rank > len(values):
        return float(values.max()) if cap_at_max else float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def run_rolling_conformal(resid_abs, pred, target, h, init_history,
                          adaptive: bool, alpha_target=ALPHA_TARGET,
                          gamma=GAMMA, cal_window=CAL_WINDOW,
                          issue_dates=None, target_dates=None, init_pending=None):
    """
    ローリング残差窓に基づくコンフォーマル予測。

    adaptive=True  -> ACI（alphaをオンライン更新）
    adaptive=False -> RollingFixed（alphaは固定。窓の更新のみ）

    フィードバック遅延の扱い（重要）:
      時刻 i で発行した区間の当否は、対象日 i+h の実現値が判明する i+h 時点まで分からない。
      そこで発行済み区間をキューに積み、実際の対象日が到来したら取り出して
      「そのとき実際に発行した（下限クリップ後の）区間が実現値を含んだか」で被覆判定し、
      その結果だけで alpha を更新する。現在時刻の q で過去の残差を再評価することはしない。
    """
    n = len(resid_abs)
    alpha_t = float(alpha_target)
    alpha_path = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    history = list(init_history)
    pending = deque(init_pending or [])
    if not history:
        raise ValueError("Calibration history must not be empty")
    if (issue_dates is None) != (target_dates is None):
        raise ValueError("Both issue_dates and target_dates are required")
    issued = np.arange(n) if issue_dates is None else pd.DatetimeIndex(issue_dates)
    realized = np.arange(n) + h if target_dates is None else pd.DatetimeIndex(target_dates)
    if len(issued) != n or len(realized) != n or any(t <= i for i, t in zip(issued, realized)):
        raise ValueError("Each target date must follow its issue date")

    for i in range(n):
        # Today's closing data is already available to the point model; release
        # resolved residuals before issuing today's interval, even across gaps.
        while pending and pending[0]["date"] <= issued[i]:
            done = pending.popleft()
            history.append(done["resid"])
            if adaptive and "lo" in done:
                covered = done["lo"] <= done["y"] <= done["hi"]
                err = 0.0 if covered else 1.0
                alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - err), ALPHA_MIN, ALPHA_MAX))
        alpha_path[i] = alpha_t
        q = conformal_quantile(history[-cal_window:], alpha_t, cap_at_max=True)

        lo = max(float(pred[i]) - q, 0.0)  # VIXは非負なので下限をクリップ
        hi = float(pred[i]) + q
        lower[i], upper[i] = lo, hi

        # 発行した区間そのものを保存（判定は必ずこのクリップ後の区間で行う）
        pending.append({"lo": lo, "hi": hi, "y": float(target[i]),
                        "resid": float(resid_abs[i]), "date": realized[i]})

    return lower, upper, alpha_path


def run_fixed_split(pred, q_fixed):
    """C. 較正期間で1度だけ算出した分位点を固定して使う（以後更新しない）。"""
    lower = np.clip(np.asarray(pred, dtype=float) - q_fixed, 0.0, None)
    upper = np.asarray(pred, dtype=float) + q_fixed
    return lower, upper


def run_naive_baseline(vix_now, q_naive):
    """D. モデルを使わない無条件区間（persistence残差の較正期間分位点、固定）。"""
    lower = np.clip(np.asarray(vix_now, dtype=float) - q_naive, 0.0, None)
    upper = np.asarray(vix_now, dtype=float) + q_naive
    return lower, upper


# =====================================================================
# ブロックブートストラップ（手法差の不確実性）
# =====================================================================
def block_bootstrap_mean_diff(a, b, block_len: int, n_boot: int = N_BOOT,
                              seed: int = BOOT_SEED, family_size: int = 1) -> dict:
    """
    対応のある2系列 a, b（各時点のスコア）の平均差 mean(a)-mean(b) について、
    moving block bootstrap で95%信頼区間を推定する。

    ブロック長は「目的変数の重複構造（h営業日）と残差の自己相関」を壊さない長さが必要。
    本分析では block_len = max(21, 2h) を用いる（h=1→21営業日≒1ヶ月、h=21→42営業日）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    if n < block_len * 2:
        raise ValueError("ブロック長に対して標本が短すぎます")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    boot = np.empty(n_boot)
    for start in range(0, n_boot, 100):
        size = min(100, n_boot - start)
        starts = rng.integers(0, n - block_len + 1, size=(size, n_blocks))
        idx = starts[:, :, None] + np.arange(block_len)[None, None, :]
        boot[start:start + size] = d[idx.reshape(size, -1)[:, :n]].mean(axis=1)

    lo, hi = np.quantile(boot, [0.025, 0.975])
    tail = 0.05 / (2 * family_size)
    adj_lo, adj_hi = np.quantile(boot, [tail, 1 - tail])
    return {
        "point_diff": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "significant": bool(adj_lo > 0 or adj_hi < 0),
        "ci_family_low": float(adj_lo), "ci_family_high": float(adj_hi),
        "family_size": family_size,
        "individual_confidence": 1 - 0.05 / family_size,
        "n_boot": int(n_boot),
        "block_len": int(block_len),
        "n_obs": int(n),
    }


# =====================================================================
# メイン
# =====================================================================
def _safe_savefig(fig, path, **kwargs):
    """
    図を保存する。

    注記: 開発中、出力先の絶対パスが Windows の MAX_PATH (260文字) を超えると
    PIL/matplotlib が FileNotFoundError を返す事象があった（真因はパス長であり、
    一時的なI/O競合ではない）。そのため出力ファイル名は短く保つこと。
    ここでは絶対パス長を事前に検査して、超過時に原因が分かるエラーを出す。
    """
    p = str(path)
    if len(p) > 255:
        raise OSError(
            f"出力パスが長すぎます（{len(p)}文字 > 255）。Windowsでは書き込みに失敗します: {p}"
        )
    fig.savefig(p, **kwargs)


def main(h: int, tag: str, make_plots: bool, source: str = "snapshot",
         run_bootstrap: bool = True, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    print("=" * 70)
    print(f"VIX {h}営業日先予測のコンフォーマル予測分析 [{tag}]")
    print("=" * 70)

    if df_raw is None:  # 呼び出し側から渡された場合は再取得しない（二重読み込みの回避）
        df_raw = build_dataset(source=source)
    feat = build_features(df_raw, h)
    n = len(feat)

    cal_start = BURN_IN_DAYS
    cal_end = min(BURN_IN_DAYS + CALIBRATION_DAYS, n)
    cal_start_date = feat.index[cal_start]
    cal_end_date = feat.index[cal_end - 1]
    test_start_date = feat.index[cal_end]

    print("\n--- 期間分割（完全な時系列分割、重複なし） ---")
    print(f"Train      : {feat.index[0].date()} 〜 {feat.index[cal_start-1].date()}  (N={cal_start})")
    print(f"Calibration: {cal_start_date.date()} 〜 {cal_end_date.date()}  (N={cal_end-cal_start})")
    print(f"Test       : {test_start_date.date()} 〜 {feat.index[-1].date()}  (N={n-cal_end})")
    RESULTS.setdefault(tag, {})["periods"] = {
        "train": [str(feat.index[0].date()), str(feat.index[cal_start - 1].date())],
        "calibration": [str(cal_start_date.date()), str(cal_end_date.date())],
        "test": [str(test_start_date.date()), str(feat.index[-1].date())],
        "n_train": int(cal_start), "n_calibration": int(cal_end - cal_start),
        "n_test": int(n - cal_end),
    }

    # --- 点予測モデル別の比較 ---
    print("\n--- 点予測モデル比較（Persistence / Linear / GBM） ---")
    model_preds, model_mae = {}, {}
    test_mask = feat.index >= test_start_date
    for name in ["persistence", "linear", "gbm"]:
        preds = walk_forward_forecast(feat, name, h)
        model_preds[name] = preds
        mae = float((feat.loc[test_mask, "target"] - preds.loc[test_mask]).abs().mean())
        model_mae[name] = mae
        print(f"  {name:12s}: Test期間MAE = {mae:.3f} VIXポイント")
    RESULTS[tag]["model_mae"] = model_mae

    # --- 4手法 × 3モデル ---
    block_len = max(21, 2 * h)
    method_results, per_obs_scores, gbm_test = {}, {}, None

    for model_name in ["persistence", "linear", "gbm"]:
        d = feat.copy()
        d["pred"] = model_preds[model_name]
        d = d.dropna(subset=["pred"])
        d["resid_abs"] = (d["target"] - d["pred"]).abs()

        cal = d[(d.index >= cal_start_date) & (d.index <= cal_end_date)]
        test = d[d.index >= test_start_date].copy()
        if len(cal) < 30 or len(test) < 30:
            continue

        # 較正期間の残差のうち、Test開始時点で実現済みのものだけを初期履歴に使う（先読み防止）。
        # 行 i の target は i+h 時点で実現するため、末尾 h 件は Test 期間に食い込む。
        known = cal["target_date"] <= test.index[0]
        cal_resid_known = cal.loc[known, "resid_abs"].values
        persist_cal_resid = (feat.loc[cal.index, "target"] - feat.loc[cal.index, "VIX"]).abs()
        persist_cal_known = persist_cal_resid.loc[known].values

        q_fixed = conformal_quantile(cal_resid_known, ALPHA_TARGET)
        q_naive = conformal_quantile(persist_cal_known, ALPHA_TARGET)
        cal_pending = [{"date": row["target_date"], "resid": row["resid_abs"]}
                       for _, row in cal.loc[~known].iterrows()]

        # A. ACI / B. RollingFixed
        for method, adaptive in [("aci", True), ("rolling_fixed", False)]:
            lo, hi, apath = run_rolling_conformal(
                test["resid_abs"].values, test["pred"].values, test["target"].values,
                h, init_history=list(cal_resid_known), adaptive=adaptive,
                issue_dates=test.index, target_dates=test["target_date"], init_pending=cal_pending,
            )
            test[f"{method}_lower"], test[f"{method}_upper"] = lo, hi
            if method == "aci":
                test["alpha_t"] = apath

        # C. FixedSplit
        lo, hi = run_fixed_split(test["pred"].values, q_fixed)
        test["fixed_split_lower"], test["fixed_split_upper"] = lo, hi

        # D. NaiveBaseline
        lo, hi = run_naive_baseline(feat.loc[test.index, "VIX"].values, q_naive)
        test["naive_lower"], test["naive_upper"] = lo, hi

        res, scores = {}, {}
        print(f"\n  [{model_name}] Test期間 N={len(test)}  "
              f"(q_fixed={q_fixed:.3f}, q_naive={q_naive:.3f})")
        for m in METHODS:
            r = evaluate_interval(test["target"].values, test[f"{m}_lower"].values,
                                  test[f"{m}_upper"].values, label=f"{model_name}/{m}")
            res[m] = r
            scores[m] = {
                "winkler": winkler_score(test["target"].values, test[f"{m}_lower"].values,
                                         test[f"{m}_upper"].values),
                "covered": ((test["target"].values >= test[f"{m}_lower"].values)
                            & (test["target"].values <= test[f"{m}_upper"].values)).astype(float),
            }
            print(f"    {METHOD_LABELS[m]:52s}: coverage={r['coverage']*100:5.2f}%  "
                  f"width={r['mean_width']:6.3f}  Winkler={r['winkler_score']:7.3f}")

        method_results[model_name] = res
        per_obs_scores[model_name] = scores
        if model_name == "gbm":
            gbm_test = test

    RESULTS[tag]["method_comparison"] = method_results

    if gbm_test is None:
        raise RuntimeError("GBMの評価結果が得られませんでした")

    # --- 手法差の不確実性（ブロックブートストラップ） ---
    if run_bootstrap:
        print(f"\n--- 手法差の不確実性（moving block bootstrap, B={N_BOOT}, "
              f"block_len={block_len}） ---")
        boot_results = {}
        pairs = [("aci", "rolling_fixed"), ("aci", "fixed_split"), ("aci", "naive"),
                 ("rolling_fixed", "fixed_split")]
        for model_name in ["linear", "gbm"]:
            if model_name not in per_obs_scores:
                continue
            boot_results[model_name] = {}
            print(f"  [{model_name}]")
            for m1, m2 in pairs:
                w = block_bootstrap_mean_diff(
                    per_obs_scores[model_name][m1]["winkler"],
                    per_obs_scores[model_name][m2]["winkler"], block_len, family_size=COMPARISON_FAMILY_SIZE)
                c = block_bootstrap_mean_diff(
                    per_obs_scores[model_name][m1]["covered"],
                    per_obs_scores[model_name][m2]["covered"], block_len, family_size=COMPARISON_FAMILY_SIZE)
                boot_results[model_name][f"{m1}_vs_{m2}"] = {"winkler_diff": w, "coverage_diff": c}
                verdict = "差を確認" if w["significant"] else "明確な差なし"
                print(f"    Winkler {m1} - {m2}: {w['point_diff']:+7.3f} "
                      f"[補正CI {w['ci_family_low']:+7.3f}, {w['ci_family_high']:+7.3f}]  {verdict}")
                verdict_c = "差を確認" if c["significant"] else "明確な差なし"
                print(f"    Coverage {m1} - {m2}: {c['point_diff']*100:+6.2f}pt "
                      f"[補正CI {c['ci_family_low']*100:+6.2f}, {c['ci_family_high']*100:+6.2f}]  {verdict_c}")
        RESULTS[tag]["bootstrap"] = boot_results

    # --- 相場局面別カバレッジ ---
    print("\n--- 相場局面別（VIXフェーズ別）カバレッジ（GBMベース） ---")
    phase_results = {}
    for phase in VIX_PHASE_LABELS:
        sub = gbm_test[gbm_test["VIX_Phase"] == phase]
        if len(sub) < 10:
            continue
        row = {m: evaluate_interval(sub["target"].values, sub[f"{m}_lower"].values,
                                    sub[f"{m}_upper"].values, label=f"{phase}/{m}")
               for m in METHODS}
        phase_results[phase] = row
        print(f"  {phase} (N={len(sub)}):")
        for m, r in row.items():
            print(f"    {m:14s}: coverage={r['coverage']*100:5.1f}% ({r['n_covered']}/{r['n']})  "
                  f"width={r['mean_width']:6.3f}  Winkler={r['winkler_score']:7.3f}")
    RESULTS[tag]["phase_coverage"] = phase_results

    # --- 急変前後のカバレッジ ---
    print("\n--- 急変（Panic突入）前後21営業日のカバレッジ（GBMベース） ---")
    episodes = find_vix_episodes(df_raw, "4_Panic (>40)", gap_days=63)
    qualifying = [(s, e) for s, e in episodes if s >= gbm_test.index[0]]
    print(f"  Test期間内のPanic突入エピソード（独立性は仮定しない）: {len(qualifying)}個")

    def _window_stats(sub, m):
        y = sub["target"].values
        lo_, hi_ = sub[f"{m}_lower"].values, sub[f"{m}_upper"].values
        cov = (y >= lo_) & (y <= hi_)
        return {"n": int(len(sub)), "n_covered": int(cov.sum()),
                "coverage": float(cov.mean()),
                "winkler": float(winkler_score(y, lo_, hi_).mean())}

    pre_post_rows = []
    for s, e in qualifying:
        pos = gbm_test.index.searchsorted(s)
        if pos >= len(gbm_test):
            continue
        pre = gbm_test.iloc[max(0, pos - 21):pos]
        post = gbm_test.iloc[pos:min(len(gbm_test), pos + 21)]
        if len(pre) < 5 or len(post) < 5:
            continue
        row = {"episode_start": str(gbm_test.index[pos].date())}
        for m in METHODS:
            row[f"{m}_pre"] = _window_stats(pre, m)
            row[f"{m}_post"] = _window_stats(post, m)
        pre_post_rows.append(row)
        print(f"  {row['episode_start']}: " + "  ".join(
            f"{m}={row[f'{m}_pre']['n_covered']}/{row[f'{m}_pre']['n']}→"
            f"{row[f'{m}_post']['n_covered']}/{row[f'{m}_post']['n']}" for m in METHODS))

    if pre_post_rows:
        avg = {}
        for m in METHODS:
            avg[m] = {
                "pre_coverage": float(np.mean([r[f"{m}_pre"]["coverage"] for r in pre_post_rows])),
                "post_coverage": float(np.mean([r[f"{m}_post"]["coverage"] for r in pre_post_rows])),
                "pre_winkler": float(np.mean([r[f"{m}_pre"]["winkler"] for r in pre_post_rows])),
                "post_winkler": float(np.mean([r[f"{m}_post"]["winkler"] for r in pre_post_rows])),
                "n_episodes": len(pre_post_rows),
            }
            print(f"  [{len(pre_post_rows)}エピソード平均] {m:14s}: "
                  f"pre={avg[m]['pre_coverage']*100:.1f}%  post={avg[m]['post_coverage']*100:.1f}%")
        RESULTS[tag]["crisis_pre_post"] = {"episodes": pre_post_rows, "average": avg}

    # --- 予測基準時点 vs 対象日基準での再集計（h=21のpre/post逆転の検証） ---
    print("\n--- 対象日(target date)基準で揃えた急変前後カバレッジ（GBMベース） ---")
    if len(gbm_test) > h:
        aligned = gbm_test.copy()  # exact dates retained before Treasury rows are dropped
        aligned_rows = []
        for s, e in qualifying:
            pre = aligned[(aligned["target_date"] < s)
                          & (aligned["target_date"] >= s - pd.Timedelta(days=45))]
            post = aligned[(aligned["target_date"] >= s)
                           & (aligned["target_date"] < s + pd.Timedelta(days=45))]
            if len(pre) < 5 or len(post) < 5:
                continue
            row = {"episode_start": str(s.date())}
            for m in METHODS:
                row[f"{m}_pre"] = _window_stats(pre, m)
                row[f"{m}_post"] = _window_stats(post, m)
            aligned_rows.append(row)
        if aligned_rows:
            aligned_avg = {}
            for m in METHODS:
                aligned_avg[m] = {
                    "pre_coverage": float(np.mean([r[f"{m}_pre"]["coverage"] for r in aligned_rows])),
                    "post_coverage": float(np.mean([r[f"{m}_post"]["coverage"] for r in aligned_rows])),
                    "n_episodes": len(aligned_rows),
                }
                print(f"  {m:14s}: pre={aligned_avg[m]['pre_coverage']*100:.1f}%  "
                      f"post={aligned_avg[m]['post_coverage']*100:.1f}%  "
                      f"({len(aligned_rows)}エピソード、対象日が危機前後45暦日に入る予測)")
            RESULTS[tag]["crisis_pre_post_target_aligned"] = {
                "episodes": aligned_rows, "average": aligned_avg}

    if make_plots:
        _make_plots(gbm_test, qualifying, h)

    return gbm_test


def _make_plots(test, episodes, h):
    FIGURES_DIR.mkdir(exist_ok=True)
    test = test.set_index("target_date", drop=False)

    # --- 実測 vs ACI区間 / 時点ごとの区間幅 / alpha_t ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(test.index, test["target"], color="black", linewidth=0.6, label="Actual VIX at target date")
    axes[0].fill_between(test.index, test["aci_lower"], test["aci_upper"],
                         color="steelblue", alpha=0.25, label="ACI 90% interval")
    for s, _ in episodes:
        axes[0].axvline(s, color="red", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("VIX")
    axes[0].set_title(f"VIX {h}-day-ahead: actual vs ACI 90% interval (Test period only)")
    axes[0].legend(loc="upper left", fontsize=9)

    for m, color, lab in [("aci", "steelblue", "A. ACI"),
                          ("rolling_fixed", "green", "B. RollingFixed"),
                          ("fixed_split", "orange", "C. FixedSplit"),
                          ("naive", "gray", "D. Naive")]:
        axes[1].plot(test.index, test[f"{m}_upper"] - test[f"{m}_lower"],
                     linewidth=0.8, color=color, alpha=0.85, label=lab)
    for s, _ in episodes:
        axes[1].axvline(s, color="red", linestyle="--", alpha=0.35)
    axes[1].set_ylabel("Interval width (VIX points)")
    axes[1].set_title("Interval width over time — width alone is not evidence of quality "
                      "(see Winkler score)")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(test.index, test["alpha_t"], color="darkred", linewidth=0.8)
    axes[2].axhline(ALPHA_TARGET, color="gray", linestyle=":")
    for s, _ in episodes:
        axes[2].axvline(s, color="red", linestyle="--", alpha=0.35)
    axes[2].set_ylabel("alpha_t (ACI only)")
    axes[2].set_title("ACI alpha at issuance, indexed by target date")
    plt.tight_layout()
    _safe_savefig(fig, FIGURES_DIR / "conformal_aci_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n図を figures/conformal_aci_overview.png に保存しました")

    # --- カバレッジ×幅の効率フロンティア ---
    fig2, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"aci": "steelblue", "rolling_fixed": "green",
              "fixed_split": "orange", "naive": "gray"}
    for m in METHODS:
        y = test["target"].values
        lo_, hi_ = test[f"{m}_lower"].values, test[f"{m}_upper"].values
        cov = ((y >= lo_) & (y <= hi_)).mean()
        width = (hi_ - lo_).mean()
        ws = winkler_score(y, lo_, hi_).mean()
        ax.scatter(width, cov * 100, s=130, color=colors[m], zorder=5,
                   label=f"{METHOD_LABELS[m].split('(')[0].strip()} (Winkler={ws:.2f})")
        ax.annotate(m, (width, cov * 100), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.axhline(90, color="red", linestyle=":", alpha=0.6, label="target coverage 90%")
    ax.set_xlabel("Mean interval width (VIX points; narrower is better, all else equal)")
    ax.set_ylabel("Realized coverage % (closer to 90% is better)")
    ax.set_title("Coverage-width tradeoff (GBM, Test period)\n"
                 "High coverage from a wide interval is not free — compare Winkler score")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    _safe_savefig(fig2, FIGURES_DIR / "conformal_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("図を figures/conformal_efficiency.png に保存しました")

    # --- 危機ズーム ---
    onsets = {"2008_GFC": pd.Timestamp("2008-09-29"), "2020_COVID": pd.Timestamp("2020-02-28")}
    available = {k: v for k, v in onsets.items() if test.index[0] <= v <= test.index[-1]}
    if available:
        fig3, axes3 = plt.subplots(1, len(available), figsize=(7.5 * len(available), 5.5))
        if len(available) == 1:
            axes3 = [axes3]
        for ax3, (name, onset) in zip(axes3, available.items()):
            pos = test.index.searchsorted(onset)
            sub = test.iloc[max(0, pos - 21):min(len(test), pos + 42)]
            ax3.plot(sub.index, sub["target"], color="black", linewidth=1.3, label="Actual VIX at target date")
            ax3.fill_between(sub.index, sub["aci_lower"], sub["aci_upper"],
                             color="steelblue", alpha=0.25, label="A. ACI")
            ax3.fill_between(sub.index, sub["rolling_fixed_lower"], sub["rolling_fixed_upper"],
                             color="green", alpha=0.18, label="B. RollingFixed")
            ax3.plot(sub.index, sub["fixed_split_lower"], color="orange",
                     linestyle="--", alpha=0.8, label="C. FixedSplit")
            ax3.plot(sub.index, sub["fixed_split_upper"], color="orange",
                     linestyle="--", alpha=0.8)
            ax3.plot(sub.index, sub["naive_lower"], color="gray", linestyle=":",
                     alpha=0.7, label="D. Naive")
            ax3.plot(sub.index, sub["naive_upper"], color="gray", linestyle=":", alpha=0.7)
            ax3.axvline(onset, color="red", linestyle="--", alpha=0.6, label="Panic onset")
            ax3.set_ylabel("VIX")
            ax3.set_title(f"{name} (h={h})")
            ax3.legend(fontsize=8)
        plt.tight_layout()
        _safe_savefig(fig3, FIGURES_DIR / "conformal_crisis_zoom.png",
                      dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print("図を figures/conformal_crisis_zoom.png に保存しました")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["snapshot", "online"], default="snapshot")
    args = parser.parse_args()

    main(h=1, tag="h1_primary", make_plots=True, source=args.source)
    main(h=21, tag="h21_robustness", make_plots=False, source=args.source)

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "results_05_conformal_vix.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n結果を {out_path.relative_to(DATA_DIR.parent)} に保存しました")
