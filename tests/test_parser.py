"""Tests for TelecomLogParser."""

import os
import pytest

from rag_system.parser import TelecomLogParser, LogRecord


@pytest.fixture
def parser():
    return TelecomLogParser()


@pytest.fixture
def sample_logs_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data", "logs")


# ── Parsing tests ──────────────────────────────────────────────────────────


class TestParseText:
    def test_parses_egate_log(self, parser):
        text = "18:34:08.417 18:34:08.417 ACR: UEC-1: UE4: Failure (code 4) while applying RRCReconfiguration message"
        records = parser.parse_text(text, "log1.txt")
        assert len(records) >= 1
        rec = records[0]
        assert isinstance(rec, LogRecord)
        assert rec.source == "log1.txt"
        assert rec.line_number == 1

    def test_parses_uec_error(self, parser):
        text = "ERROR!! 18:34:08'417\"426|75bb06c0 rfma_impl.cpp[80]:VIP: UEC-1: UE4: Failure (code 4)"
        records = parser.parse_text(text, "log2.txt")
        assert len(records) >= 1
        assert records[0].log_level == "ERROR"

    def test_parses_rain_log(self, parser):
        text = '7f ASP-2834-2-cp_ue <2026-04-15T18:34:08.586830Z> 62-cp_ue INF/cp_ue/ConcreteUeSaMethods.cpp:271 [ueIdCu:4] Trigger ue release'
        records = parser.parse_text(text, "log3.txt")
        assert len(records) >= 1
        assert "ue release" in records[0].message.lower() or "trigger" in records[0].message.lower()

    def test_empty_input(self, parser):
        records = parser.parse_text("", "empty.txt")
        assert records == []

    def test_line_numbers_are_correct(self, parser):
        text = "line one\n18:34:08.417 18:34:08.417 ACR: error something\nline three"
        records = parser.parse_text(text, "test.txt")
        for rec in records:
            assert rec.line_number >= 1


class TestParseFolder:
    def test_parses_sample_logs(self, parser, sample_logs_dir):
        if not os.path.exists(sample_logs_dir):
            pytest.skip("Sample logs not available")
        records = parser.parse_folder(sample_logs_dir)
        assert len(records) > 0
        assert all(isinstance(r, LogRecord) for r in records)

    def test_multiple_sources(self, parser, sample_logs_dir):
        if not os.path.exists(sample_logs_dir):
            pytest.skip("Sample logs not available")
        records = parser.parse_folder(sample_logs_dir)
        sources = {r.source for r in records}
        assert len(sources) >= 2, "Should parse logs from multiple files"


class TestLogRecord:
    def test_to_dict(self, parser):
        text = "18:34:08.417 18:34:08.417 ACR: UEC-1: UE4: Failure (code 4) while applying RRCReconfiguration message"
        records = parser.parse_text(text, "test.txt")
        assert len(records) >= 1
        d = records[0].to_dict()
        assert "source" in d
        assert "message" in d
        assert d["source"] == "test.txt"
