# -*- coding: utf-8 -*-
"""
単一のデータ取得・特徴量生成パイプライン。
このリポジトリの全ての分析はこのモジュールの build_dataset() が返す DataFrame のみを参照する。

データソースの方針（再現性）:
  - 既定は `data/vix_sp500_tnx_raw.csv`（スナップショット）からの読み込み。
    → ネットワークなしで、README記載の数値をそのまま再現できる。
  - `build_dataset(source="online")` を明示した場合のみ yfinance から再取得する。
    → 再取得結果でスナップショットを更新したい場合は `refresh_snapshot=True` を指定する。
  - どちらを使ったかは必ず標準出力へ表示する。

注意（設計上の限界を明示）:
  - 6ヶ月先リターンは `shift(-126)` による「126営業日先」であり、正確な暦日6ヶ月後ではない。
  - VIXとS&P500の両方が揃う営業日のみを使用する。
  - 米10年債(^TNX)のみ休場の日が存在するため、金利を使う分析だけ標本が小さくなる。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- パス定義（カレントディレクトリに依存しない） ---
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
DOCS_DIR = REPO_ROOT / "docs"
SNAPSHOT_PATH = DATA_DIR / "vix_sp500_tnx_raw.csv"

START = "1990-01-01"
END = "2024-01-01"
SNAPSHOT_FETCH_DATE = "2026-09-08"  # スナップショットを取得した日

RAW_COLUMNS = ["SP500", "VIX", "Treasury_10Y"]

VIX_PHASE_BINS = [0, 20, 30, 40, np.inf]
VIX_PHASE_LABELS = ["1_Normal (<20)", "2_Alert (20-30)", "3_Fear (30-40)", "4_Panic (>40)"]

ROLLING_WINDOW = 63   # VIX_Z算出用の窓（営業日）
HORIZON_DAYS = 126    # 6ヶ月先リターンの営業日換算（暦日6ヶ月ではない）


def _fetch_close_online(ticker: str) -> pd.Series:
    """1銘柄の終値をyfinanceから取得する（列名は明示的にticker名にする）。"""
    import yfinance as yf  # オンライン取得時のみ import（テストを外部依存なしで走らせるため）

    raw = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):  # yfinanceのMultiIndex対策
        s = s.iloc[:, 0]
    s = s.rename(ticker)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s


def _load_raw_online() -> pd.DataFrame:
    sp500 = _fetch_close_online("^GSPC").rename("SP500")
    vix = _fetch_close_online("^VIX").rename("VIX")
    tnx = _fetch_close_online("^TNX").rename("Treasury_10Y")
    return pd.concat([sp500, vix, tnx], axis=1)


def _load_raw_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    """保存済みスナップショットから生データ3列のみを読み込む（派生列は必ず再計算する）。"""
    if not path.exists():
        raise FileNotFoundError(
            f"スナップショットが見つかりません: {path}\n"
            f"オンライン取得する場合は build_dataset(source='online') を明示してください。"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"スナップショットに必要な列がありません: {missing}")
    return df[RAW_COLUMNS].copy()


def validate_raw(df: pd.DataFrame, verbose: bool = True) -> dict:
    """生データの妥当性を検査し、検査結果を辞書で返す（テストからも呼ぶ）。"""
    report = {}

    # 日付インデックス: 昇順・重複なし
    if not df.index.is_monotonic_increasing:
        raise ValueError("日付インデックスが昇順ではありません")
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        raise ValueError(f"重複した日付が {n_dup} 件あります")
    report["n_duplicate_dates"] = n_dup

    # 必須列
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必須列がありません: {missing}")

    # 値域
    if not df["VIX"].dropna().between(5, 100).mean() > 0.99:
        raise ValueError("VIXの値域が想定外です")
    if not df["SP500"].dropna().gt(0).all():
        raise ValueError("SP500に非正値があります")
    tnx = df["Treasury_10Y"].dropna()
    if len(tnx) and not tnx.between(0, 20).mean() > 0.99:
        raise ValueError("Treasury_10Yの値域が想定外です（単位が%であることを想定）")

    report["n_rows"] = int(len(df))
    report["start"] = str(df.index.min().date())
    report["end"] = str(df.index.max().date())
    report["n_missing"] = {c: int(df[c].isna().sum()) for c in RAW_COLUMNS}

    if verbose:
        print(f"  行数={report['n_rows']}  期間={report['start']}〜{report['end']}  "
              f"重複日付={n_dup}件")
        print(f"  欠損: {report['n_missing']}  "
              f"（Treasury_10Yのみの欠損は債券市場だけ休場の日に対応）")
    return report


def build_dataset(source: str = "snapshot", refresh_snapshot: bool = False,
                  verbose: bool = True) -> pd.DataFrame:
    """
    SP500・VIX・10年債利回りを読み込み、分析に必要な特徴量を全て付与した単一のDataFrameを返す。

    source: "snapshot"（既定、オフライン再現）または "online"（yfinanceから再取得）
    refresh_snapshot: source="online" のときにスナップショットCSVを更新するか
    """
    if source not in ("snapshot", "online"):
        raise ValueError(f"source は 'snapshot' か 'online': {source}")

    if source == "online":
        if verbose:
            print(f"[データソース] yfinanceから再取得 ({START}〜{END})")
        raw = _load_raw_online()
    else:
        if verbose:
            print(f"[データソース] スナップショット {SNAPSHOT_PATH.name} "
                  f"(取得日 {SNAPSHOT_FETCH_DATE})")
        raw = _load_raw_snapshot()

    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    df = raw.dropna(subset=["SP500", "VIX"]).copy()  # 両方が揃う営業日のみ使用

    if verbose:
        validate_raw(df, verbose=True)
    else:
        validate_raw(df, verbose=False)

    if source == "online" and refresh_snapshot:
        DATA_DIR.mkdir(exist_ok=True)
        df[RAW_COLUMNS].to_csv(SNAPSHOT_PATH)
        if verbose:
            print(f"[スナップショット更新] {SNAPSHOT_PATH}")

    # --- 派生列は常にここで再計算する（スナップショット側の派生列は使わない） ---

    # 目的変数: 126営業日先のリターン（暦日6ヶ月後ではない）。未来値はここでのみ使用。
    df["SP500_Future_6M"] = df["SP500"].shift(-HORIZON_DAYS)
    df["Return_6M"] = df["SP500_Future_6M"] / df["SP500"] - 1

    # VIXフェーズ分類（水準ベース）
    df["VIX_Phase"] = pd.cut(df["VIX"], bins=VIX_PHASE_BINS, labels=VIX_PHASE_LABELS)

    # VIXのローリングZスコア（過去方向のみ）
    df["VIX_Mean_63d"] = df["VIX"].rolling(ROLLING_WINDOW).mean()
    df["VIX_Std_63d"] = df["VIX"].rolling(ROLLING_WINDOW).std()
    df["VIX_Z"] = (df["VIX"] - df["VIX_Mean_63d"]) / df["VIX_Std_63d"]

    # 金利の定常化: 日次差分と63日変化
    df["Treasury_10Y_Diff1"] = df["Treasury_10Y"].diff(1)
    df["Treasury_10Y_Diff63"] = df["Treasury_10Y"].diff(ROLLING_WINDOW)

    return df


def analysis_frame(df: pd.DataFrame, require_cols) -> pd.DataFrame:
    """指定した列が揃っている行だけを残した解析用フレームを返す（用途ごとに標本が変わることを明示）。"""
    return df.dropna(subset=list(require_cols)).copy()


def find_vix_episodes(df: pd.DataFrame, phase_label: str = "4_Panic (>40)", gap_days: int = 63):
    """
    指定したVIXフェーズの日付を、gap_days営業日以上の空白で区切って独立エピソードにクラスタリングする。

    gap_days=63（約3ヶ月）は、6ヶ月先リターンの重複期間の半分にあたり、
    「別の危機」と見なせる最低限の間隔として設定した探索的な値である（感度分析は未実施）。

    戻り値: [(開始日, 終了日), ...]
    """
    dates = df.index[df["VIX_Phase"] == phase_label]
    if len(dates) == 0:
        return []
    pos = {d: i for i, d in enumerate(df.index)}
    episodes = []
    start = prev = dates[0]
    for d in dates[1:]:
        if pos[d] - pos[prev] > gap_days:
            episodes.append((start, prev))
            start = d
        prev = d
    episodes.append((start, prev))
    return episodes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="データパイプラインの実行")
    parser.add_argument("--source", choices=["snapshot", "online"], default="snapshot",
                        help="データソース（既定: snapshot＝オフライン再現）")
    parser.add_argument("--refresh-snapshot", action="store_true",
                        help="source=online のときにスナップショットCSVを更新する")
    args = parser.parse_args()

    data = build_dataset(source=args.source, refresh_snapshot=args.refresh_snapshot)
    print("\n取得完了。全体形状:", data.shape)
    print("期間:", data.index.min().date(), "〜", data.index.max().date())
    print("\n列ごとの欠損数:")
    print(data.isna().sum())
