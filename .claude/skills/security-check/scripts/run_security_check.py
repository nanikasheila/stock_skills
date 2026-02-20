#!/usr/bin/env python3
"""Git リポジトリのセキュリティ・個人情報チェッカー.

追跡ファイル内およびコミット履歴のメタデータから
個人情報・機密情報の漏洩リスクを検出する。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Finding:
    """検出結果."""

    category: str
    severity: Severity
    message: str
    location: str = ""
    detail: str = ""
    remediation: str = ""


@dataclass
class CheckResult:
    """チェック結果の集約."""

    findings: list[Finding] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Optional[str] = None) -> str:
    """Git コマンドを実行して stdout を返す."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd or _get_repo_root(),
            timeout=60,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_repo_root() -> str:
    """リポジトリのルートパスを取得."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else os.getcwd()


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# メールアドレス（個人用ドメイン）
_PERSONAL_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.co.jp", "yahoo.com", "hotmail.com",
    "outlook.com", "outlook.jp", "icloud.com", "me.com",
    "mac.com", "live.com", "live.jp", "msn.com",
    "protonmail.com", "proton.me",
]

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# API キー・トークンパターン
_SECRET_PATTERNS = [
    (re.compile(r"xai-[a-zA-Z0-9]{20,}"), "Grok/xAI API Key"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI API Key"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "GitHub Personal Access Token"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "GitHub OAuth Token"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{22,}"), "GitHub Fine-grained PAT"),
    (re.compile(r"glpat-[a-zA-Z0-9\-_]{20,}"), "GitLab PAT"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS Access Key ID"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "Google API Key"),
    (re.compile(r"neo4j\+s?://[^:]+:[^@]+@"), "Neo4j Connection URI with password"),
]

# 電話番号パターン（日本）— ハイフン区切り必須で誤検出を防止
_PHONE_PATTERNS = [
    re.compile(r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b"),  # 固定電話 (03-1234-5678)
    re.compile(r"\b0[789]0-\d{4}-\d{4}\b"),  # 携帯電話 (090-1234-5678)
]

# 住所パターン
_ADDRESS_PATTERN = re.compile(r"〒?\d{3}-\d{4}")

# ローカルホスト名パターン（MacBook-Pro, DESKTOP-XXX など）
_HOSTNAME_PATTERN = re.compile(
    r"(?:MacBook|iMac|DESKTOP|LAPTOP|PC)[a-zA-Z0-9\-_.]*\.local",
    re.IGNORECASE,
)

# パスワード代入パターン（実際の値が入っている場合）
_PASSWORD_ASSIGN_PATTERN = re.compile(
    r"""(?:password|passwd|pwd)\s*[=:]\s*["'](?!password|changeme|example|xxx|your)[a-zA-Z0-9!@#$%^&*]{6,}["']""",
    re.IGNORECASE,
)

# noreply は除外
_NOREPLY_PATTERN = re.compile(r"noreply@", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------

def check_git_authors(result: CheckResult) -> None:
    """Git コミット履歴の著者名・メールアドレスをチェック."""
    output = _run_git(["log", "--all", "--format=%an|%ae|%cn|%ce"])
    if not output:
        return

    seen = set()
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        author_name, author_email, committer_name, committer_email = parts[:4]

        for name, email, role in [
            (author_name, author_email, "Author"),
            (committer_name, committer_email, "Committer"),
        ]:
            key = f"{role}:{name}:{email}"
            if key in seen:
                continue
            seen.add(key)

            # noreply は無視
            if _NOREPLY_PATTERN.search(email):
                continue

            # 個人メールアドレスチェック
            domain = email.split("@")[-1].lower() if "@" in email else ""
            if domain in _PERSONAL_EMAIL_DOMAINS:
                result.findings.append(Finding(
                    category="個人メールアドレス",
                    severity=Severity.MEDIUM,
                    message=f"{role} に個人メールアドレスが使用されています",
                    location=f"git log ({role})",
                    detail=f"{name} <{email}>",
                    remediation="git filter-repo で書き換えるか、今後のコミットは git config user.email で匿名化",
                ))

            # ローカルホスト名がメールに含まれる
            if _HOSTNAME_PATTERN.search(email):
                result.findings.append(Finding(
                    category="ローカルホスト名",
                    severity=Severity.MEDIUM,
                    message=f"{role} メールにローカルホスト名が含まれています",
                    location=f"git log ({role})",
                    detail=f"{name} <{email}>",
                    remediation="git filter-repo で書き換え",
                ))

            # 実名っぽい名前のチェック（日本語文字を含む場合）
            if re.search(r"[\u3040-\u9fff]", name):
                result.findings.append(Finding(
                    category="実名(日本語)",
                    severity=Severity.HIGH,
                    message=f"{role} に日本語の実名が含まれています",
                    location=f"git log ({role})",
                    detail=f"{name} <{email}>",
                    remediation="git filter-repo --mailmap で匿名化",
                ))


def check_tracked_files(result: CheckResult) -> None:
    """追跡ファイルの内容をチェック."""
    repo_root = _get_repo_root()

    # 追跡ファイル一覧
    file_list = _run_git(["ls-tree", "-r", "HEAD", "--name-only"])
    if not file_list:
        return

    for filepath in file_list.splitlines():
        # バイナリファイルはスキップ
        ext = Path(filepath).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                   ".ttf", ".eot", ".pdf", ".zip", ".tar", ".gz", ".bin",
                   ".pyc", ".pyo", ".so", ".dll", ".exe"}:
            continue

        full_path = os.path.join(repo_root, filepath)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        _check_file_content(result, filepath, content)


def _check_file_content(result: CheckResult, filepath: str, content: str) -> None:
    """ファイル内容の個人情報・機密情報をチェック."""
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        # メールアドレス（コメントや文字列リテラル内）
        for match in _EMAIL_PATTERN.finditer(line):
            email = match.group()
            if _NOREPLY_PATTERN.search(email):
                continue
            # example.com 等のダミーは無視
            domain = email.split("@")[-1].lower()
            if domain in {"example.com", "example.org", "test.com", "localhost"}:
                continue
            if domain in _PERSONAL_EMAIL_DOMAINS:
                result.findings.append(Finding(
                    category="個人メールアドレス",
                    severity=Severity.HIGH,
                    message="ファイル内に個人メールアドレスがハードコードされています",
                    location=f"{filepath}:{line_num}",
                    detail=email,
                    remediation="環境変数化またはプレースホルダーに置換",
                ))

        # APIキー・トークン
        for pattern, name in _SECRET_PATTERNS:
            if pattern.search(line):
                result.findings.append(Finding(
                    category="機密情報",
                    severity=Severity.HIGH,
                    message=f"{name} が検出されました",
                    location=f"{filepath}:{line_num}",
                    detail=f"パターン: {name}",
                    remediation=".env ファイルに移動し、.gitignore に追加",
                ))

        # パスワード代入
        if _PASSWORD_ASSIGN_PATTERN.search(line):
            result.findings.append(Finding(
                category="機密情報",
                severity=Severity.HIGH,
                message="パスワードがハードコードされている可能性があります",
                location=f"{filepath}:{line_num}",
                detail=line.strip()[:80],
                remediation="環境変数化し、.env に移動",
            ))

        # 電話番号
        for pattern in _PHONE_PATTERNS:
            m = pattern.search(line)
            if m:
                # 数値が単にバージョン番号・金額・テストデータ等でないか簡易チェック
                stripped = line.strip()
                matched_str = m.group()

                # 除外キーワード
                skip_keywords = [
                    "version", "port", "#", "python", "cap", "cash",
                    "volume", "close", "price", "amount", "target",
                    "monthly", "add", "market_cap", "min_market",
                    "help=", "例:", "example", "1000000", "test",
                ]
                if any(kw in stripped.lower() for kw in skip_keywords):
                    continue

                # 数字のみ（セパレータなし）の長い数列は金額の可能性
                digits_only = re.sub(r"[-() ]", "", matched_str)
                if len(digits_only) > 8:
                    continue

                # 前後がアルファベットや数字に隣接していたら電話番号ではない可能性
                start_pos = m.start()
                if start_pos > 0 and line[start_pos - 1].isalnum():
                    continue

                result.findings.append(Finding(
                    category="電話番号",
                    severity=Severity.MEDIUM,
                    message="電話番号パターンが検出されました",
                    location=f"{filepath}:{line_num}",
                    detail=stripped[:60],
                    remediation="個人電話番号であれば削除",
                ))

        # 住所（郵便番号）
        if _ADDRESS_PATTERN.search(line):
            result.findings.append(Finding(
                category="住所",
                severity=Severity.MEDIUM,
                message="日本の郵便番号パターンが検出されました",
                location=f"{filepath}:{line_num}",
                detail=line.strip()[:60],
                remediation="個人住所であれば削除",
            ))


def check_gitignore(result: CheckResult) -> None:
    """.gitignore の重要パターンをチェック."""
    repo_root = _get_repo_root()
    gitignore_path = os.path.join(repo_root, ".gitignore")

    if not os.path.exists(gitignore_path):
        result.findings.append(Finding(
            category="設定不備",
            severity=Severity.HIGH,
            message=".gitignore ファイルが存在しません",
            location=".gitignore",
            remediation=".gitignore を作成し、機密ファイルを除外",
        ))
        return

    with open(gitignore_path, "r") as f:
        content = f.read()

    # 必須パターンのチェック
    required_patterns = {
        ".env": "環境変数ファイル（APIキー等）",
        "*.pyc": "Pythonバイトコード",
        "__pycache__": "Python キャッシュ",
    }

    for pattern, desc in required_patterns.items():
        if pattern not in content:
            result.findings.append(Finding(
                category="設定不備",
                severity=Severity.MEDIUM,
                message=f".gitignore に {pattern} が含まれていません",
                location=".gitignore",
                detail=desc,
                remediation=f".gitignore に '{pattern}' を追加",
            ))

    # .env ファイルが追跡されていないか確認
    tracked = _run_git(["ls-tree", "-r", "HEAD", "--name-only"])
    if tracked:
        for line in tracked.splitlines():
            basename = Path(line).name
            if basename == ".env" or basename.endswith(".env.local"):
                result.findings.append(Finding(
                    category="機密情報",
                    severity=Severity.HIGH,
                    message=f".env ファイルが Git で追跡されています",
                    location=line,
                    remediation="git rm --cached で追跡を解除し、.gitignore に追加",
                ))


def check_sensitive_files_in_history(result: CheckResult) -> None:
    """過去のコミット履歴に機密ファイルが残っていないかチェック."""
    sensitive_patterns = [
        ("*.csv", "CSVデータ（個人投資データの可能性）"),
        ("*.env", "環境変数ファイル"),
        ("*.pem", "証明書/秘密鍵"),
        ("*.key", "秘密鍵"),
        ("*id_rsa*", "SSH鍵"),
    ]

    for pattern, desc in sensitive_patterns:
        output = _run_git(["log", "--all", "--oneline", "--diff-filter=A", "--", pattern])
        if output:
            lines = output.splitlines()
            for line in lines[:3]:  # 最大3件
                result.findings.append(Finding(
                    category="履歴内の機密ファイル",
                    severity=Severity.LOW,
                    message=f"過去のコミットに {pattern} が追加されたことがあります",
                    location=f"git log: {line.strip()[:60]}",
                    detail=desc,
                    remediation="git filter-repo で完全削除（必要に応じて）",
                ))


def check_os_username_in_files(result: CheckResult) -> None:
    """OSユーザー名がファイルに含まれていないかチェック."""
    # 現在のOSユーザー名
    username = os.environ.get("USERNAME") or os.environ.get("USER", "")
    if not username or len(username) < 3:
        return

    # ホームディレクトリのパスパターン
    home_patterns = [
        f"C:\\\\Users\\\\{username}",
        f"C:/Users/{username}",
        f"/Users/{username}",
        f"/home/{username}",
    ]

    repo_root = _get_repo_root()
    file_list = _run_git(["ls-tree", "-r", "HEAD", "--name-only"])
    if not file_list:
        return

    for filepath in file_list.splitlines():
        ext = Path(filepath).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".pyc"}:
            continue

        full_path = os.path.join(repo_root, filepath)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue

        for pattern in home_patterns:
            if pattern.lower() in content.lower():
                result.findings.append(Finding(
                    category="OSユーザー名",
                    severity=Severity.MEDIUM,
                    message="ファイル内にOSユーザー名を含むパスがあります",
                    location=filepath,
                    detail=f"パターン: {pattern}",
                    remediation="相対パスまたは環境変数に置換",
                ))
                break  # 同一ファイルで1回のみ


def check_hardcoded_paths(result: CheckResult) -> None:
    """ハードコードされた絶対パスをチェック."""
    repo_root = _get_repo_root()
    file_list = _run_git(["ls-tree", "-r", "HEAD", "--name-only"])
    if not file_list:
        return

    path_pattern = re.compile(
        r"(?:/Users/[a-zA-Z0-9_]+/|/home/[a-zA-Z0-9_]+/|C:\\Users\\[a-zA-Z0-9_]+\\)",
        re.IGNORECASE,
    )

    for filepath in file_list.splitlines():
        ext = Path(filepath).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".pyc"}:
            continue

        full_path = os.path.join(repo_root, filepath)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    match = path_pattern.search(line)
                    if match:
                        result.findings.append(Finding(
                            category="ハードコードされたパス",
                            severity=Severity.MEDIUM,
                            message="ユーザー固有の絶対パスがハードコードされています",
                            location=f"{filepath}:{line_num}",
                            detail=match.group()[:60],
                            remediation="相対パスまたは環境変数に置換",
                        ))
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_SEVERITY_ICONS = {
    Severity.HIGH: "🔴",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "ℹ️",
}


def format_report(result: CheckResult, verbose: bool = False) -> str:
    """結果をテキストレポートに整形."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("🔒 Git セキュリティ・個人情報チェック結果")
    lines.append(f"   実行日時: {result.checked_at}")
    lines.append("=" * 60)
    lines.append("")

    # サマリー
    total = len(result.findings)
    if total == 0:
        lines.append("✅ 問題は検出されませんでした。")
        return "\n".join(lines)

    lines.append(f"検出件数: {total} 件")
    lines.append(f"  🔴 HIGH: {result.high_count}  🟡 MEDIUM: {result.medium_count}  "
                 f"🔵 LOW: {result.low_count}  ℹ️ INFO: {result.info_count}")
    lines.append("")

    # カテゴリ別グルーピング
    categories: dict[str, list[Finding]] = {}
    for f in result.findings:
        categories.setdefault(f.category, []).append(f)

    for category, findings in categories.items():
        lines.append(f"── {category} ({len(findings)}件) " + "─" * 30)
        for f in findings:
            icon = _SEVERITY_ICONS[f.severity]
            lines.append(f"  {icon} [{f.severity.value}] {f.message}")
            if f.location:
                lines.append(f"     場所: {f.location}")
            if f.detail and verbose:
                lines.append(f"     詳細: {f.detail}")
            if f.remediation and verbose:
                lines.append(f"     対策: {f.remediation}")
        lines.append("")

    # 推奨アクション
    if result.high_count > 0:
        lines.append("─" * 60)
        lines.append("⚠️ HIGH レベルの問題が検出されました。早急に対処してください。")

        # 具体的な対策をリスト
        remediations = set()
        for f in result.findings:
            if f.severity == Severity.HIGH and f.remediation:
                remediations.add(f.remediation)
        if remediations:
            lines.append("")
            lines.append("推奨アクション:")
            for i, r in enumerate(sorted(remediations), 1):
                lines.append(f"  {i}. {r}")

    return "\n".join(lines)


def format_json(result: CheckResult) -> str:
    """結果をJSON形式で出力."""
    data = {
        "checked_at": result.checked_at,
        "summary": {
            "total": len(result.findings),
            "high": result.high_count,
            "medium": result.medium_count,
            "low": result.low_count,
            "info": result.info_count,
        },
        "findings": [
            {
                "category": f.category,
                "severity": f.severity.value,
                "message": f.message,
                "location": f.location,
                "detail": f.detail,
                "remediation": f.remediation,
            }
            for f in result.findings
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_checks() -> CheckResult:
    """全チェックを実行."""
    result = CheckResult()

    check_git_authors(result)
    check_tracked_files(result)
    check_gitignore(result)
    check_sensitive_files_in_history(result)
    check_os_username_in_files(result)
    check_hardcoded_paths(result)

    # 重大度順にソート
    severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    result.findings.sort(key=lambda f: severity_order[f.severity])

    return result


def main():
    parser = argparse.ArgumentParser(description="Git セキュリティ・個人情報チェッカー")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="出力形式 (default: text)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="詳細情報(detail/remediation)を表示",
    )
    args = parser.parse_args()

    result = run_all_checks()

    if args.format == "json":
        print(format_json(result))
    else:
        print(format_report(result, verbose=args.verbose))

    # HIGH があれば exit code 1
    sys.exit(1 if result.high_count > 0 else 0)


if __name__ == "__main__":
    main()
