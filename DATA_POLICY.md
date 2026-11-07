# データの方針

研究用コードと市場データの利用権は別です。このリポジトリには原データを含めません。

- **出典:** Yahoo Finance（yfinance経由）の `^GSPC`、`^VIX`、`^TNX`
- **本READMEの取得日:** 2026-09-08
- **含めないもの:** `data/vix_sp500_tnx_raw.csv`
- **含めるもの:** コード、集計結果のJSON、レポート、図

Yahoo Financeデータの再配布許諾は確認できていません。データを取得する場合は、利用者自身の用途で許諾を確認してください。

```bash
python data_pipeline.py --source online --refresh-snapshot
```

調整済みデータは後日改訂される場合があります。再取得したデータでは、READMEの数値と完全には一致しないことがあります。
