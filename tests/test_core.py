"""Unit tests for the pure analysis logic (no camera, GPU, Ollama or FastAPI needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevin_core import (SceneMemory, clean, danger_score, inject_rtsp_credentials,  # noqa: E402
                        is_small_model, match_keywords, ollama_base, safe_join, tg_escape)

KWS = ["weapon", "armed", "fire", "fight", "attack", "drag", "mask", "crowd", "break-in", "covered face"]


# ---- keyword matching ---------------------------------------------------------
def test_plain_keyword_triggers():
    assert match_keywords("A man is holding a weapon near the door.", KWS) == ["weapon"]


def test_inflections_still_match():
    assert "attack" in match_keywords("Someone attacked the cashier.", KWS)
    assert "fight" in match_keywords("Two men are fighting in the lot.", KWS)
    assert "mask" in match_keywords("A masked person enters.", KWS)


def test_whole_words_only():
    # These all fired with the old substring matching
    assert match_keywords("An unarmed guard stands by the gate.", KWS) == []
    assert match_keywords("A family sits by the fireplace.", KWS) == []
    assert match_keywords("A child holds a dragon toy.", KWS) == []
    assert match_keywords("A firefighter walks past.", KWS) == []


def test_negation_suppresses():
    assert match_keywords("No weapon is visible.", KWS) == []
    assert match_keywords("There is no sign of fire or smoke.", KWS) == []


def test_negation_in_other_clause_does_not_suppress():
    assert match_keywords("Nobody is at the counter, but a man is fighting outside.", KWS) == ["fight"]


def test_multiword_and_hyphenated_keywords():
    assert match_keywords("Signs of a break-in at the back door.", KWS) == ["break-in"]
    assert match_keywords("A person with a covered face waits.", KWS) == ["covered face"]


def test_bad_keyword_lists_are_safe():
    assert match_keywords("fire", None) == []
    assert match_keywords("fire", ["", "  "]) == []
    # Regex special characters in user keywords must not crash matching
    assert match_keywords("price is $5 (sale)", ["$5", "(sale)"]) == ["$5", "(sale)"]


# ---- danger score -------------------------------------------------------------
def test_danger_score_ranges():
    calm = danger_score([], True, 0)
    normal = danger_score([], False, 5)
    alert = danger_score(["weapon", "fight"], False, 50)
    assert 0 <= calm < normal < alert <= 1
    assert danger_score(["a"] * 20, False, 500) == 1.0


# ---- scene memory -------------------------------------------------------------
def test_baseline_learned_after_calm_frames():
    m = SceneMemory()
    for i in range(3):
        m.update("Empty parking lot.", 0.05, f"10:00:0{i}")
    assert m.baseline == "Empty parking lot."
    ctx = m.build_context("Describe.", "gemma3:4b")
    assert "NORMAL for this camera" in ctx and "WHAT JUST HAPPENED" in ctx


def test_trend_rising_and_falling():
    m = SceneMemory()
    for d in [0.1, 0.1, 0.1, 0.8, 0.9, 0.9]:
        m.update("x", d, "t")
    assert m.trend() == "rising"
    m2 = SceneMemory()
    for d in [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]:
        m2.update("x", d, "t")
    assert m2.trend() == "falling"


def test_small_model_gets_compact_prompt():
    assert is_small_model("moondream:latest")
    m = SceneMemory()
    m.baseline = "A quiet hallway."
    ctx = m.build_context("Describe.", "moondream")
    assert ctx.startswith("Normally:") and "WHAT JUST HAPPENED" not in ctx


# ---- text cleanup -------------------------------------------------------------
def test_clean_strips_preamble_and_markdown():
    assert clean("The image shows **a man** running.") == "A man running."


def test_clean_caps_sentences():
    out = clean("One. Two. Three. Four. Five.")
    assert out == "One. Two. Three."


def test_clean_removes_prompt_echo():
    prompt = "Describe who is present and what they do."
    assert clean(prompt + " A woman is reading.", prompt) == "A woman is reading."


# ---- helpers ------------------------------------------------------------------
def test_ollama_base_normalises():
    assert ollama_base("http://h:11434/api/chat") == "http://h:11434"
    assert ollama_base("http://h:11434/v1/") == "http://h:11434"
    assert ollama_base("http://h:11434/") == "http://h:11434"


def test_rtsp_credentials_are_url_encoded():
    out = inject_rtsp_credentials("rtsp://10.0.0.5:554/stream1", "admin", "p@ss:w/rd#1")
    assert out == "rtsp://admin:p%40ss%3Aw%2Frd%231@10.0.0.5:554/stream1"


def test_rtsp_existing_credentials_untouched():
    uri = "rtsp://u:p@10.0.0.5/stream"
    assert inject_rtsp_credentials(uri, "admin", "x") == uri
    assert inject_rtsp_credentials("rtsp://10.0.0.5/s", "", "x") == "rtsp://10.0.0.5/s"


def test_safe_join_blocks_traversal(tmp_path):
    base = tmp_path / "snapshots"
    base.mkdir()
    assert safe_join(str(base), "2026-01-01/cam_1.jpg") == str(base / "2026-01-01" / "cam_1.jpg")
    assert safe_join(str(base), "../secret.txt") is None
    # Absolute paths are kept inside base rather than escaping it
    assert safe_join(str(base), "/etc/passwd") == str(base / "etc" / "passwd")
    assert safe_join(str(tmp_path / "snap"), "../snapshots/x.jpg") is None  # prefix-sibling trick


def test_tg_escape():
    assert tg_escape("a <b> & c") == "a &lt;b&gt; &amp; c"
