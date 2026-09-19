"""
demo/generate_demo_assets.py
Generates local, reproducible demo artifacts using PyMuPDF and Python standard libraries.
"""
import os
import sys
import wave
import struct
import math
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import fitz  # PyMuPDF (already in requirements.txt)

DEMO_DIR = ROOT_DIR / "demo"
CORPUS_DIR = DEMO_DIR / "corpus"
SUBMISSIONS_DIR = DEMO_DIR / "submissions"
AUDIO_DIR = DEMO_DIR / "audio"
CODE_DIR = DEMO_DIR / "code"
EXPECTED_DIR = DEMO_DIR / "expected_results"

for d in [CORPUS_DIR, SUBMISSIONS_DIR, AUDIO_DIR, CODE_DIR, EXPECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Corpus Assets
# ----------------------------------------------------------------------
REF_MATERIAL_P1 = (
    "Machine Learning in Modern Data Analysis\n\n"
    "Machine learning algorithms identify patterns in historical data to forecast unseen outcomes. "
    "Rather than following rigid hand-crafted instructions, statistical models learn functional "
    "representations directly from sample observations. Deep neural network architectures have "
    "revolutionized natural language processing and computer vision by extracting hierarchical representations. "
    "Training procedures adjust parameter weights via gradient descent to minimize empirical loss across datasets. "
    "Supervised learning algorithms generalize across complex computational domains when provided with abundant training samples."
)

REF_MATERIAL_P2 = (
    "Evaluation and Generalization in Predictive Systems\n\n"
    "Supervised predictive modeling separates data into distinct training and validation partitions. "
    "Classification algorithms assign categorical labels to input feature vectors based on learned decision boundaries. "
    "Conversely, regression methodologies approximate continuous numerical target variables from multidimensional observations. "
    "Overfitting occurs when high-capacity statistical models memorize random stochastic noise rather than underlying patterns. "
    "Regularization techniques constrain model complexity to ensure robust out-of-sample generalization in real-world environments."
)

# Build demo/corpus/reference_material.pdf
pdf_path = CORPUS_DIR / "reference_material.pdf"
doc = fitz.open()
page1 = doc.new_page(width=595, height=842)  # A4
page1.insert_textbox(fitz.Rect(50, 50, 545, 780), REF_MATERIAL_P1, fontsize=12, fontname="helv")
page2 = doc.new_page(width=595, height=842)
page2.insert_textbox(fitz.Rect(50, 50, 545, 780), REF_MATERIAL_P2, fontsize=12, fontname="helv")
doc.save(str(pdf_path))
doc.close()

# Build demo/corpus/reference_material_2.txt
(CORPUS_DIR / "reference_material_2.txt").write_text(
    "Dimensionality Reduction and Unsupervised Feature Extraction\n\n"
    "Unsupervised machine learning discovers latent clusters and structural manifolds without explicit ground-truth targets. "
    "Principal component analysis identifies orthogonal vectors that maximize variance across high-dimensional feature spaces. "
    "Clustering techniques segment unstructured observations into homogeneous semantic partitions. "
    "These statistical abstractions empower automated recommendation engines, anomaly detectors, and cross-modal retrieval indices.",
    encoding="utf-8",
)

# Build demo/corpus/reference_code.py
(CORPUS_DIR / "reference_code.py").write_text(
    "def compute_dataset_metrics(records):\n"
    "    total_sum = 0\n"
    "    filtered_count = 0\n"
    "    for record in records:\n"
    "        if record > 0:\n"
    "            total_sum += record\n"
    "            filtered_count += 1\n"
    "    mean_value = total_sum / max(1, filtered_count)\n"
    "    return mean_value\n",
    encoding="utf-8",
)

# ----------------------------------------------------------------------
# 2. Submissions (Cross-Lingual & Paraphrased)
# ----------------------------------------------------------------------
# Real Hindi semantic paraphrase of Paragraph 1
HINDI_TEXT = (
    "मशीन लर्निंग एल्गोरिदम ऐतिहासिक डेटा में पैटर्न की पहचान करते हैं ताकि अनदेखे परिणामों का पूर्वानुमान लगाया जा सके। "
    "कठोर हस्तनिर्मित नियमों का पालन करने के बजाय, सांख्यिकीय मॉडल सीधे अवलोकनों से कार्यात्मक प्रतिनिधित्व सीखते हैं। "
    "गहरे तंत्रिका नेटवर्क आर्किटेक्चर ने स्तरित प्रतिनिधित्व सीखकर भाषा प्रसंस्करण में क्रांति ला दी है। "
    "डेटासेट में त्रुटि को कम करने के लिए प्रशिक्षण प्रक्रिया ग्रेडिएंट डिसेंट के माध्यम से वजन को समायोजित करती है।"
)
(SUBMISSIONS_DIR / "cross_lingual_hindi.txt").write_text(HINDI_TEXT, encoding="utf-8")

PARAPHRASE_TEXT = (
    "Computational Intelligence in Applied Data Science\n\n"
    "ML algorithms discover relationships within datasets to predict prospective conclusions. "
    "Statistical frameworks master computational abstractions from provided records instead of adhering to predetermined instructions. "
    "Deep neural architectures have altered computational linguistics by acquiring structural feature hierarchies. "
    "Optimization steps modify internal coefficients using gradient descent to lower observation error across samples. "
    "Supervised learning procedures transfer insights across multidimensional tasks when supplied with adequate sample records.\n\n"
    "Assessment Principles for Automated Predictors\n\n"
    "Supervised inference separates observations into designated training and test splits. "
    "Categorization models assign categorical classes to features according to determined decision frontiers. "
    "In contrast, regression formulations estimate numeric quantities from multivariate observations. "
    "Excessive parameterization happens whenever models memorize stochastic variance rather than true relationships. "
    "Penalization mechanisms restrict architectural complexity to support stable predictive accuracy on unseen data."
)

(SUBMISSIONS_DIR / "paraphrased_submission.txt").write_text(PARAPHRASE_TEXT, encoding="utf-8")

# Build demo/submissions/paraphrased_submission.pdf
sub_pdf_path = SUBMISSIONS_DIR / "paraphrased_submission.pdf"
sub_doc = fitz.open()
sub_page1 = sub_doc.new_page(width=595, height=842)
sub_page1.insert_textbox(fitz.Rect(50, 50, 545, 780), PARAPHRASE_TEXT[:len(PARAPHRASE_TEXT)//2], fontsize=12, fontname="helv")
sub_page2 = sub_doc.new_page(width=595, height=842)
sub_page2.insert_textbox(fitz.Rect(50, 50, 545, 780), PARAPHRASE_TEXT[len(PARAPHRASE_TEXT)//2:], fontsize=12, fontname="helv")
sub_doc.save(str(sub_pdf_path))
sub_doc.close()

# ----------------------------------------------------------------------
# 3. Audio Assets (Standard Wave Module + Script Fallback)
# ----------------------------------------------------------------------
AUDIO_SCRIPT = (
    "Machine learning algorithms identify patterns in historical data to forecast unseen outcomes. "
    "Rather than following rigid hand-crafted instructions, statistical models learn functional representations directly. "
    "Deep neural network architectures have revolutionized natural language processing by extracting hierarchical representations."
)
(AUDIO_DIR / "speech_script.txt").write_text(AUDIO_SCRIPT, encoding="utf-8")

# Generate 3-second 16-bit PCM WAV tone (valid audio container compatible with all players/Whisper)
wav_path = AUDIO_DIR / "speech_sample.wav"
sample_rate = 16000
duration_sec = 3.0
frequency = 440.0
total_frames = int(sample_rate * duration_sec)

with wave.open(str(wav_path), "w") as wf:
    wf.setnchannels(1)       # Mono
    wf.setsampwidth(2)       # 16-bit
    wf.setframerate(sample_rate)
    frames = bytearray()
    for i in range(total_frames):
        # Soft sinusoidal beep envelope
        t = float(i) / sample_rate
        envelope = math.sin(math.pi * t / duration_sec)
        sample_val = int(envelope * 16000.0 * math.sin(2.0 * math.pi * frequency * t))
        frames.extend(struct.pack("<h", max(-32768, min(32767, sample_val))))
    wf.writeframes(frames)

# ----------------------------------------------------------------------
# 4. Code Assets (AST Source vs. Suspect Refactored)
# ----------------------------------------------------------------------
SOURCE_CODE = (
    "def calculate_average(numbers):\n"
    "    total = 0\n"
    "    for number in numbers:\n"
    "        total += number\n"
    "    return total / len(numbers)\n"
)
(CODE_DIR / "source_code.py").write_text(SOURCE_CODE, encoding="utf-8")

SUSPECT_CODE = (
    "def compute_mean(values):\n"
    "    # Cosmetic variable renaming and refactored identifier names\n"
    "    accumulator = 0\n"
    "    for val in values:\n"
    "        accumulator += val\n"
    "    return accumulator / len(values)\n"
)
(CODE_DIR / "suspect_code.py").write_text(SUSPECT_CODE, encoding="utf-8")

# ----------------------------------------------------------------------
# 5. Expected Results (Qualitative Benchmarks Only)
# ----------------------------------------------------------------------
(EXPECTED_DIR / "demo_expectations.json").write_text(
    """{
  "scenario_1_corpus_ingestion": {
    "files": ["reference_material.pdf", "reference_material_2.txt"],
    "expected_status": "Indexed into FAISS & SQLite",
    "minimum_expected_chunks": 4
  },
  "scenario_2_cross_lingual": {
    "source_language": "en",
    "submission_language": "hi",
    "expected_lexical_similarity": "very_low_or_zero",
    "expected_semantic_similarity": "high_greater_than_0.70",
    "expected_match_type": "Near-Direct Translation / Semantic Theft"
  },
  "scenario_3_document_paraphrase": {
    "expected_coverage_range": "26_to_100_percent",
    "expected_risk_band": "Moderate or Severe",
    "expected_match_type": "Paraphrased Semantic Theft"
  },
  "scenario_4_code_ast": {
    "expected_structural_similarity": "greater_than_0.90",
    "expected_variable_renaming_detected": true,
    "expected_classification": "Semantic Code Logic Match (High Confidence Refactor)"
  },
  "scenario_5_audio_speech": {
    "expected_modality": "audio",
    "expected_match_type": "Cross-Modal Audio",
    "expected_matched_source": "reference_material.pdf"
  }
}
""",
    encoding="utf-8",
)

print("[✓] All demo assets successfully created inside demo/")