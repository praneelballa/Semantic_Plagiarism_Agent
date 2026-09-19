"""
demo/run_demo_check.py
Validates the demo pack environment, file presence, and syntax prior to presentations.
"""
import ast
import json
import sys
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT_DIR / "demo"

sys.path.insert(0, str(ROOT_DIR / "backend"))
import fitz  # PyMuPDF


def check_demo_pack():
    print("\n" + "=" * 50)
    print("      DEMO ENVIRONMENT & ARTIFACT CHECK       ")
    print("=" * 50)

    # 1. Corpus Check
    pdf_path = DEMO_DIR / "corpus" / "reference_material.pdf"
    txt_path = DEMO_DIR / "corpus" / "reference_material_2.txt"
    assert pdf_path.exists(), "Missing reference_material.pdf"
    assert txt_path.exists(), "Missing reference_material_2.txt"
    with fitz.open(str(pdf_path)) as doc:
        assert len(doc) >= 2, "reference_material.pdf must have at least 2 pages"
        text = doc[0].get_text()
        assert len(text) > 100, "reference_material.pdf Page 1 text empty"
    print("✓ Corpus: reference_material.pdf & reference_material_2.txt valid")

    # 2. Cross-Lingual Check
    hi_path = DEMO_DIR / "submissions" / "cross_lingual_hindi.txt"
    assert hi_path.exists(), "Missing cross_lingual_hindi.txt"
    hi_text = hi_path.read_text(encoding="utf-8")
    assert any("\u0900" <= c <= "\u097F" for c in hi_text), "Hindi text must contain Devanagari script"
    print("✓ Cross-Lingual: Hindi submission verified")

    # 3. Paraphrased Document Check
    sub_txt = DEMO_DIR / "submissions" / "paraphrased_submission.txt"
    sub_pdf = DEMO_DIR / "submissions" / "paraphrased_submission.pdf"
    assert sub_txt.exists() and sub_pdf.exists(), "Missing paraphrased submission files"
    with fitz.open(str(sub_pdf)) as doc:
        assert len(doc) >= 2, "paraphrased_submission.pdf must have at least 2 pages"
    print("✓ Document: Paraphrased TXT and PDF submissions verified")

    # 4. Code AST Check
    src_code = DEMO_DIR / "code" / "source_code.py"
    sus_code = DEMO_DIR / "code" / "suspect_code.py"
    assert src_code.exists() and sus_code.exists(), "Missing code demo files"
    tree_src = ast.parse(src_code.read_text(encoding="utf-8"))
    tree_sus = ast.parse(sus_code.read_text(encoding="utf-8"))
    assert len(tree_src.body) > 0 and len(tree_sus.body) > 0, "AST syntax check failed"
    print("✓ Code: Source and suspect Python AST syntax verified")

    # 5. Audio Check
    wav_path = DEMO_DIR / "audio" / "speech_sample.wav"
    script_path = DEMO_DIR / "audio" / "speech_script.txt"
    assert wav_path.exists() and script_path.exists(), "Missing audio file or script fallback"
    with wave.open(str(wav_path), "r") as wf:
        assert wf.getnchannels() == 1, "Audio should be mono"
        assert wf.getframerate() == 16000, "Sample rate should be 16kHz"
    print("✓ Audio: 16kHz WAV file and script fallback verified")

    # 6. Expectations
    exp_path = DEMO_DIR / "expected_results" / "demo_expectations.json"
    assert exp_path.exists(), "Missing demo_expectations.json"
    with open(exp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "scenario_2_cross_lingual" in data
    print("✓ Expectations: Schema configuration verified")

    print("-" * 50)
    print("DEMO PACK READY FOR PRESENTATION")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    check_demo_pack()