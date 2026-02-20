````skill
---
name: security-check
description: Gitリポジトリのセキュリティ・個人情報チェック。コミット履歴・追跡ファイルから個人情報・機密情報の漏洩リスクを検出する。
argument-hint: "[--verbose] [--format text|json]  例: --verbose, --format json"
allowed-tools: Bash(python3 *)
---

# セキュリティ・個人情報チェックスキル

Gitリポジトリ内の個人情報・機密情報を自動検出する。

## チェック項目

1. **Gitコミット著者情報** — 個人メールアドレス・実名・ローカルホスト名
2. **追跡ファイルの内容** — ハードコードされたメールアドレス・APIキー・パスワード・電話番号・住所
3. **.gitignore設定** — .env等の必須除外パターン
4. **過去コミット履歴** — CSV・.env・秘密鍵等の機密ファイル
5. **OSユーザー名の漏洩** — ファイル内の絶対パス
6. **ハードコードされたパス** — ユーザー固有の絶対パス

## 実行

```bash
python3 .claude/skills/security-check/scripts/run_security_check.py $ARGUMENTS
```

## 引数

| 引数 | 説明 |
|:---|:---|
| `--verbose` / `-v` | 詳細情報（detail/remediation）を表示 |
| `--format text` | テキスト形式で出力（デフォルト） |
| `--format json` | JSON形式で出力 |

## 出力

検出された問題をカテゴリ別・重要度別に表示する。

重要度レベル:
- 🔴 **HIGH** — 即座に対処が必要（APIキー漏洩、個人メールのハードコード等）
- 🟡 **MEDIUM** — 対処推奨（コミット履歴のメール、ハードコードされたパス等）
- 🔵 **LOW** — 注意（過去コミットのCSVファイル等）
- ℹ️ **INFO** — 参考情報

HIGHレベルの問題がある場合、exit code 1 を返す。

````
