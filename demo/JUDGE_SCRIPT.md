# 60–90 Second Live Demonstration Pitch

"Good morning/afternoon, judges.

Traditional plagiarism detection systems like Turnitin and MOSS rely heavily on string matching, n-grams, and superficial word frequencies. The moment an author swaps synonyms, translates an essay into another language, transcribes a spoken lecture, or renames variables in code, traditional detectors break down completely.

Our project is a **Multimodal and Cross-Lingual Semantic Plagiarism Detection Agent**. It does not merely look for identical words—it projects meaning into a shared geometric latent space.

Let us demonstrate in five immediate scenarios:

1. **First, Corpus Ingestion:** We index two academic papers on machine learning into our local FAISS vector database. Notice that chunk metadata, page boundaries, and token offsets are preserved in SQLite.
2. **Second, Cross-Lingual Evasion:** Here is an English passage, and here is its direct translation in Hindi. Notice the metrics: lexical overlap is 0.0%. Yet, our multilingual MiniLM model reveals a high semantic similarity of over 0.75, immediately tagging it as a *Near-Direct Translation*.
3. **Third, Paraphrased Documents:** When we submit a completely reworded PDF essay, our system calculates actual word-token coverage rather than diluting the score with a document-wide average. The system flags 50%+ coverage and highlights the exact overlapping passages side-by-side with source page references.
4. **Fourth, AST Code Logic Theft:** When a student renames variables, changes function names, and modifies comments, traditional string-diff tools see two different programs. Our Abstract Syntax Tree de-aliaser strips cosmetic identifiers, reveals identical canonical structures, and flags *Variable Renaming Detected* with 100% structural fidelity.
5. **Finally, Audio Speech Theft:** If someone transcribes a recorded spoken lecture, OpenAI Whisper converts the speech into timestamped intervals, matches the ideas against the text corpus, and cites the exact audio timestamp—such as 00:04 to 00:28—corresponding to Page 1 of our reference document.

**Instead of asking whether two submissions use the same words, our system asks whether they communicate or implement the same underlying information.**

Thank you."