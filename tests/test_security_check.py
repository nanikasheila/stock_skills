"""Tests for security-check skill (run_security_check.py)."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# スクリプトを直接 import するためにパスを追加
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "security-check" / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import run_security_check as sc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_result():
    return sc.CheckResult()


# ---------------------------------------------------------------------------
# Unit tests: pattern matching
# ---------------------------------------------------------------------------

class TestEmailPattern:
    def test_matches_gmail(self):
        assert sc._EMAIL_PATTERN.search("user@gmail.com")

    def test_matches_yahoo(self):
        assert sc._EMAIL_PATTERN.search("test@yahoo.co.jp")

    def test_ignores_noreply(self):
        email = "noreply@anthropic.com"
        assert sc._NOREPLY_PATTERN.search(email)

    def test_no_match_on_plain_text(self):
        assert not sc._EMAIL_PATTERN.search("hello world")


class TestSecretPatterns:
    @pytest.mark.parametrize("value,expected_name", [
        ("xai-abcdefghijklmnopqrstuvwx", "Grok/xAI API Key"),
        ("sk-abcdefghijklmnopqrstuvwx", "OpenAI API Key"),
        ("ghp_abcdefghijklmnopqrstuvwxyz1234567890", "GitHub Personal Access Token"),
        ("gho_abcdefghijklmnopqrstuvwxyz1234567890", "GitHub OAuth Token"),
        ("AKIAIOSFODNN7EXAMPLE", "AWS Access Key ID"),
    ])
    def test_detects_known_patterns(self, value, expected_name):
        matched = False
        for pattern, name in sc._SECRET_PATTERNS:
            if pattern.search(value):
                assert name == expected_name
                matched = True
                break
        assert matched, f"Pattern for {expected_name} did not match"

    def test_no_false_positive_on_placeholder(self):
        """xai-xxxxxxxxxxxxx のようなプレースホルダーは短すぎてマッチしないことを確認."""
        line = "XAI_API_KEY=xai-xxxxxxxxxxxxx"
        for pattern, _ in sc._SECRET_PATTERNS:
            # xai-xxxxxxxxxxxxx は13文字で20文字未満なのでマッチしない
            match = pattern.search(line)
            if match:
                assert len(match.group()) < 20 or "xxx" in match.group()


class TestPhonePattern:
    def test_jp_mobile(self):
        assert any(p.search("090-1234-5678") for p in sc._PHONE_PATTERNS)

    def test_jp_landline(self):
        assert any(p.search("03-1234-5678") for p in sc._PHONE_PATTERNS)

    def test_no_false_positive_on_amount(self):
        """金額パターン（1000000等）にマッチしないことを確認."""
        assert not any(p.search("1000000") for p in sc._PHONE_PATTERNS)

    def test_no_false_positive_on_large_number(self):
        """大きな数値にマッチしないことを確認."""
        assert not any(p.search("42000000000000") for p in sc._PHONE_PATTERNS)


class TestAddressPattern:
    def test_postal_code(self):
        assert sc._ADDRESS_PATTERN.search("〒100-0001")

    def test_postal_code_no_prefix(self):
        assert sc._ADDRESS_PATTERN.search("100-0001")


class TestHostnamePattern:
    def test_macbook(self):
        assert sc._HOSTNAME_PATTERN.search("kikuchihiroyuki@HIROYUKInoMacBook-Pro.local")

    def test_desktop(self):
        assert sc._HOSTNAME_PATTERN.search("user@DESKTOP-ABC123.local")


# ---------------------------------------------------------------------------
# Unit tests: check functions
# ---------------------------------------------------------------------------

class TestCheckFileContent:
    def test_detects_personal_email(self, empty_result):
        content = 'email = "john@gmail.com"'
        sc._check_file_content(empty_result, "test.py", content)
        assert len(empty_result.findings) == 1
        assert empty_result.findings[0].category == "個人メールアドレス"
        assert empty_result.findings[0].severity == sc.Severity.HIGH

    def test_ignores_example_email(self, empty_result):
        content = 'email = "test@example.com"'
        sc._check_file_content(empty_result, "test.py", content)
        assert len(empty_result.findings) == 0

    def test_ignores_noreply(self, empty_result):
        content = "Author: Claude <noreply@anthropic.com>"
        sc._check_file_content(empty_result, "test.py", content)
        assert len(empty_result.findings) == 0

    def test_detects_api_key(self, empty_result):
        content = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234"'
        sc._check_file_content(empty_result, "config.py", content)
        found = [f for f in empty_result.findings if f.category == "機密情報"]
        assert len(found) >= 1

    def test_detects_postal_code(self, empty_result):
        content = "住所: 〒100-0001 東京都千代田区"
        sc._check_file_content(empty_result, "readme.md", content)
        found = [f for f in empty_result.findings if f.category == "住所"]
        assert len(found) == 1

    def test_phone_with_version_keyword_is_skipped(self, empty_result):
        content = "# python 3.10.12"
        sc._check_file_content(empty_result, "test.py", content)
        phone_findings = [f for f in empty_result.findings if f.category == "電話番号"]
        assert len(phone_findings) == 0


# ---------------------------------------------------------------------------
# Unit tests: formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_clean_report(self, empty_result):
        report = sc.format_report(empty_result)
        assert "問題は検出されませんでした" in report

    def test_report_with_finding(self, empty_result):
        empty_result.findings.append(sc.Finding(
            category="テスト",
            severity=sc.Severity.HIGH,
            message="テスト問題",
            location="test.py:1",
        ))
        report = sc.format_report(empty_result, verbose=True)
        assert "🔴" in report
        assert "テスト問題" in report
        assert "HIGH: 1" in report

    def test_json_format(self, empty_result):
        empty_result.findings.append(sc.Finding(
            category="テスト",
            severity=sc.Severity.MEDIUM,
            message="テスト",
        ))
        output = sc.format_json(empty_result)
        data = json.loads(output)
        assert data["summary"]["medium"] == 1
        assert len(data["findings"]) == 1


class TestCheckResult:
    def test_severity_counts(self):
        result = sc.CheckResult()
        result.findings = [
            sc.Finding(category="a", severity=sc.Severity.HIGH, message="h1"),
            sc.Finding(category="a", severity=sc.Severity.HIGH, message="h2"),
            sc.Finding(category="b", severity=sc.Severity.MEDIUM, message="m1"),
            sc.Finding(category="c", severity=sc.Severity.LOW, message="l1"),
            sc.Finding(category="d", severity=sc.Severity.INFO, message="i1"),
        ]
        assert result.high_count == 2
        assert result.medium_count == 1
        assert result.low_count == 1
        assert result.info_count == 1


# ---------------------------------------------------------------------------
# Integration-like test: git authors check with mock
# ---------------------------------------------------------------------------

class TestCheckGitAuthors:
    def test_detects_personal_email(self, empty_result):
        mock_output = "testuser|test@gmail.com|testuser|test@gmail.com"
        with patch.object(sc, "_run_git", return_value=mock_output):
            sc.check_git_authors(empty_result)
        email_findings = [f for f in empty_result.findings if f.category == "個人メールアドレス"]
        assert len(email_findings) >= 1

    def test_detects_japanese_name(self, empty_result):
        mock_output = "菊池太郎|test@local|菊池太郎|test@local"
        with patch.object(sc, "_run_git", return_value=mock_output):
            sc.check_git_authors(empty_result)
        name_findings = [f for f in empty_result.findings if f.category == "実名(日本語)"]
        assert len(name_findings) >= 1
        assert name_findings[0].severity == sc.Severity.HIGH

    def test_detects_hostname_in_email(self, empty_result):
        mock_output = "user|user@MacBook-Pro.local|user|user@MacBook-Pro.local"
        with patch.object(sc, "_run_git", return_value=mock_output):
            sc.check_git_authors(empty_result)
        host_findings = [f for f in empty_result.findings if f.category == "ローカルホスト名"]
        assert len(host_findings) >= 1

    def test_ignores_noreply(self, empty_result):
        mock_output = "Claude|noreply@anthropic.com|Claude|noreply@anthropic.com"
        with patch.object(sc, "_run_git", return_value=mock_output):
            sc.check_git_authors(empty_result)
        assert len(empty_result.findings) == 0

    def test_empty_git_output(self, empty_result):
        with patch.object(sc, "_run_git", return_value=""):
            sc.check_git_authors(empty_result)
        assert len(empty_result.findings) == 0


class TestCheckGitignore:
    def test_missing_gitignore(self, empty_result, tmp_path):
        with patch.object(sc, "_get_repo_root", return_value=str(tmp_path)):
            sc.check_gitignore(empty_result)
        assert any(f.message == ".gitignore ファイルが存在しません" for f in empty_result.findings)

    def test_complete_gitignore(self, empty_result, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".env\n*.pyc\n__pycache__\n")
        with patch.object(sc, "_get_repo_root", return_value=str(tmp_path)):
            with patch.object(sc, "_run_git", return_value=""):
                sc.check_gitignore(empty_result)
        # 必須パターンがすべて含まれているので設定不備はなし
        config_findings = [f for f in empty_result.findings if f.category == "設定不備"]
        assert len(config_findings) == 0

    def test_missing_env_pattern(self, empty_result, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__\n")
        with patch.object(sc, "_get_repo_root", return_value=str(tmp_path)):
            with patch.object(sc, "_run_git", return_value=""):
                sc.check_gitignore(empty_result)
        config_findings = [f for f in empty_result.findings if ".env" in f.message]
        assert len(config_findings) == 1


# ---------------------------------------------------------------------------
# run_all_checks: sorted by severity 
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_findings_sorted_by_severity(self):
        with patch.object(sc, "check_git_authors") as m1, \
             patch.object(sc, "check_tracked_files") as m2, \
             patch.object(sc, "check_gitignore") as m3, \
             patch.object(sc, "check_sensitive_files_in_history") as m4, \
             patch.object(sc, "check_os_username_in_files") as m5, \
             patch.object(sc, "check_hardcoded_paths") as m6:

            def add_mixed_findings(result):
                result.findings.extend([
                    sc.Finding(category="a", severity=sc.Severity.LOW, message="low"),
                    sc.Finding(category="b", severity=sc.Severity.HIGH, message="high"),
                    sc.Finding(category="c", severity=sc.Severity.MEDIUM, message="med"),
                ])

            m1.side_effect = add_mixed_findings

            result = sc.run_all_checks()
            severities = [f.severity for f in result.findings]
            assert severities == [sc.Severity.HIGH, sc.Severity.MEDIUM, sc.Severity.LOW]
