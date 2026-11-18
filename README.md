# nlp-assignment

# Lab Assignment 1: NLP Tool and Framework Review
**Course:** MSc in Artificial Intelligence - Natural Language Processing (NLP)
**Student:** Ezira Mantegaftot (GSR0232)
**Topic:** Python Libraries and Frameworks for Text Preprocessing & Language Modeling

---

## 1. Overview of the Python Implementation (`nlp_tools_review.py`)

This lab module provides an empirical and conceptual demonstration of the tools surveyed in Section III of the theoretical paper:

1. **General-Purpose NLP (NLTK vs. spaCy)**:
   - Demonstrates pedagogical/exploratory regex tokenization (NLTK) vs. compiled production pipelines (spaCy).
   - Contrasts architectural speed, memory usage, and the absence of pre-trained Amharic models in spaCy.
2. **Amharic-Specific Preprocessing Frameworks**:
   - `fidel-tools`: Implementation of canonical homophone normalization (`ሀ/ሐ/ኀ/ኃ`, `ሰ/ሠ`, `አ/ዐ`, `ጸ/ፀ`).
   - `amseg`: Implementation of sentence boundary detection respecting Ethiopic punctuation (`።`) and abbreviations (`ዓ.ም.`, `ት/ቤት`, `ዶ/ር`) without false splits.
   - `amharic-text-processor`: End-to-end cleaning pipeline, stopword removal, and rule-based light stemming (highlighting risks of over-stemming Semitic roots).
3. **Language Modeling Paradigms (N-Grams vs. Transformers)**:
   - Traditional statistical Bigram LM with Laplace ($+1$) smoothing, showing perplexity calculation under the Markov assumption.
   - Modern Subword Tokenizer simulation (BPE/WordPiece) showing how agglutinative Amharic words (e.g., `በቤተመጻሕፍታችንም`) are decomposed into morphemic subwords to prevent Out-of-Vocabulary (OOV) expansion.
   - Hugging Face Transformers integration hook for multilingual models (`Davlan/afro-xlmr-base`, `xlm-roberta-base`).
4. **Comparative Synthesis Matrix**:
   - Prints an IEEE-styled comparative table contrasting speed, memory, low-resource viability, and target deployment domains.

---

## 2. How to Run

Run the script from PowerShell or command prompt:

```powershell
python nlp_tools_review.py
```

*Note: The script is fully self-contained using Python's standard library with zero external dependencies required to run the core simulation. If `nltk`, `spacy`, or `transformers` are installed in your environment, it will automatically leverage their native modules.*