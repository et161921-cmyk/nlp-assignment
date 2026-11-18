"""
Assignment 1: Theoretical Paper & NLP Tool/Framework Review
MSc in Artificial Intelligence - Natural Language Processing (NLP)
Author: Ezira Mantegaftot (GSR0232)

Comprehensive Empirical Lab Suite:
Evaluating NLP Frameworks & Libraries against the Ambiguity Challenge
(Lexical, Syntactic/SOV, Pragmatic) for English and Amharic
"""

import re
import math
from collections import defaultdict, Counter

# ==============================================================================
# SECTION II & III EMPIRICAL BENCHMARKS (Exact Sentences from Paper)
# ==============================================================================

PAPER_EXAMPLES = {
    "lexical_en": "The crane flew over the construction site where a heavy crane was working.",
    "lexical_am_1": "ልጁ ምግቡን በላ።",                   # በላ = ate (verb)
    "lexical_am_2": "በላ ንገረኝ፣ ዝም አትበል።",             # በላ = speak up / come on (interjection)
    "lexical_am_3": "እዚያ ሰው አለ።",                     # አለ = exists / there is
    "lexical_am_4": "እውነቱን አለ።",                     # አለ = he spoke / said

    "syntactic_en": "I saw the man with a camera",       # PP attachment ambiguity
    "syntactic_am": "ልጁን በመኪና ያዘው",                  # caught using car vs. inside car
    "order_svo_en": "The boy ate the food.",             # English SVO
    "order_sov_am": "ልጁ ምግቡን በላ።",                   # Amharic SOV: Subject - Object - Verb

    "clitic_am_1": "በቤታችን",                            # በ- (prep) + ቤት (noun) + -አችን (possessive)
    "clitic_am_2": "አልሄደም",                           # አል- (neg) + ሄድ (root) + -ኤም (neg suffix)

    "pragmatic_en": "It is cold in here",                # Literal temp vs. "close the window"
    "pragmatic_am": "በሩ ክፍት ነው"                        # Literal state vs. "close the door"
}


# ==============================================================================
# 1. AMHARIC PREPROCESSING UTILITIES (amseg, fidel-tools, amharic-text-proc)
# ==============================================================================

class FidelToolsNormalizer:
    """
    fidel-tools (Abraham, 2022): Script Normalization & Cleaning.
    Normalizes Ethiopic homophone graphemes to their canonical representatives
    (ሀ/ሐ/ኀ/ኃ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ) to reduce vocabulary sparsity.
    """
    def __init__(self):
        self.homophone_map = {
            'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
            'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
            'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
            'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
            'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
        }

    def normalize(self, text: str) -> str:
        return "".join(self.homophone_map.get(char, char) for char in text)


class AmsegSegmenter:
    """
    amseg (Yimam et al., 2021): Tokenization & Sentence Segmentation.
    Understands Ethiopic punctuation (Arat Neteb '።', Netela Neteb '፡', Serez '፣')
    and preserves abbreviations (ዓ.ም., ት/ቤት, ዶ/ር) without false splits.
    """
    def __init__(self):
        self.abbreviations = [
            'ዓ.ም.', 'ዓ/ም', 'ት/ቤት', 'ት/ክፍል', 'ዶ/ር', 'ፕ/ር', 'ወ/ሮ', 'ወ/ሪት',
            'ክ/ከተማ', 'መ/ቤት', 'ጠ/ሚኒስትር', 'ገ/ገንዘብ', 'ኢ/ር'
        ]

    def tokenize_sentences(self, text: str) -> list:
        protected_text = text
        placeholder_map = {}
        for idx, abbr in enumerate(self.abbreviations):
            placeholder = f"__ABBR_{idx}__"
            if abbr in protected_text:
                protected_text = protected_text.replace(abbr, placeholder)
                placeholder_map[placeholder] = abbr

        raw_sentences = re.split(r'[።!?]+', protected_text)
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            if s_clean:
                for ph, orig in placeholder_map.items():
                    s_clean = s_clean.replace(ph, orig)
                sentences.append(s_clean + "።")
        return sentences

    def tokenize_words(self, text: str) -> list:
        cleaned = re.sub(r'[።፣፤፥፡\-–—()\[\]"\'«»]', ' ', text)
        return [w for w in cleaned.split() if w]


class AmharicTextProcessor:
    """
    amharic-text-processor (Seyoum, 2023): End-to-End Cleaning & Stemming.
    Bundles homophone normalization, punctuation/digit filtering, stop-word removal,
    and rule-based prefix/suffix stripping.
    """
    def __init__(self):
        self.normalizer = FidelToolsNormalizer()
        self.stopwords = {
            'ነው', 'ናት', 'ነበሩ', 'እንደ', 'ላይ', 'ውስጥ', 'ስለ', 'ጋር', 'እና', 'ወይም',
            'ግን', 'ከ', 'በ', 'ወደ', 'እስከ', 'ብቻ', 'ሁሉ', 'ይህ', 'ይህን', 'እነዚህ',
            'ያ', 'ያን', 'እነዚያ', 'አለ', 'ነበር', 'ሆነ'
        }
        self.prefixes = ['የ', 'በ', 'ለ', 'ከ', 'እንደ', 'ስለ', 'እየ', 'አል']
        self.suffixes = ['ኦች', 'ዎች', 'ኡ', 'ዋ', 'አችን', 'አቸው', 'ኤ', 'ህ', 'ሽ', 'ም', 'ና', 'ኤም']

    def light_stem(self, word: str) -> str:
        stemmed = word
        for p in self.prefixes:
            if stemmed.startswith(p) and len(stemmed) > len(p) + 2:
                stemmed = stemmed[len(p):]
                break
        for s in self.suffixes:
            if stemmed.endswith(s) and len(stemmed) > len(s) + 2:
                stemmed = stemmed[:-len(s)]
                break
        return stemmed

    def clean_and_preprocess(self, text: str) -> list:
        text = self.normalizer.normalize(text)
        text = re.sub(r'[a-zA-Z0-9\d]+', ' ', text)
        tokens = re.findall(r'[\u1200-\u137F]+', text)
        processed = [self.light_stem(t) for t in tokens if t not in self.stopwords]
        return processed


# ==============================================================================
# 2. LANGUAGE MODELING ENGINES (N-Grams vs. Transformer Subwords)
# ==============================================================================

class TraditionalNGramLM:
    """Statistical N-Gram Language Model with Laplace (Add-1) Smoothing (nltk.lm)."""
    def __init__(self, n: int = 2):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, tokenized_sentences: list):
        for sent in tokenized_sentences:
            padded = ['<s>'] * (self.n - 1) + sent + ['</s>']
            self.vocab.update(sent)
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1

    def probability(self, word: str, context: tuple) -> float:
        ngram = context + (word,)
        v = len(self.vocab) + 1
        return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + v)


class SubwordTokenizerDemo:
    """Simulates BPE/SentencePiece subword tokenization for Semitic agglutinative morphemes."""
    def __init__(self):
        self.subword_vocab = {
            "ልጁ", "ልጁን", "ምግቡን", "በላ", "ሰላም", "አለ", "በ", "ቤት", "አችን", "አል", "ሄድ", "ኤም",
            "መኪና", "ያዘው", "በሩ", "ክፍት", "ነው"
        }

    def tokenize(self, word: str) -> list:
        subwords = []
        start = 0
        while start < len(word):
            matched = False
            for end in range(len(word), start, -1):
                piece = word[start:end]
                prefix = piece if start == 0 else "##" + piece
                if piece in self.subword_vocab:
                    subwords.append(prefix)
                    start = end
                    matched = True
                    break
            if not matched:
                subwords.append(word[start] if start == 0 else "##" + word[start])
                start += 1
        return subwords


# ==============================================================================
# 3. COMPREHENSIVE FRAMEWORK-BY-FRAMEWORK EVALUATION ACROSS AMBIGUITIES
# ==============================================================================

def evaluate_frameworks_on_paper():
    print("\n" + "="*80)
    print("EMPIRICAL EVALUATION: MAPPING FRAMEWORKS TO THE PAPER'S AMBIGUITY CHALLENGES")
    print("="*80)

    # --------------------------------------------------------------------------
    # 1. NLTK
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 1: NLTK (Natural Language Toolkit)")
    print("#"*70)
    print("Role: Academic research & pedagogical sandbox (Pure Python).")
    print("\n[A. Lexical Ambiguity: English 'Crane' & Amharic 'በላ', 'አለ']")
    print("  * English: NLTK offers WordNet (nltk.corpus.wordnet) for synset lookup,")
    print("    enabling Lesk algorithm disambiguation of 'Crane' (bird vs. machine).")
    print("  * Amharic: NLTK has ZERO native Ge'ez lexicons or WordNet for Amharic.")
    print(f"    Raw split of '{PAPER_EXAMPLES['lexical_am_1']}':")
    nltk_split_am = re.findall(r'[\w]+|[።]', PAPER_EXAMPLES['lexical_am_1'])
    print(f"    -> Output tokens: {nltk_split_am}")
    print("    -> Failure: 'በላ' (ate) vs. 'በላ' (speak up) remain identical discrete strings.")

    print("\n[B. Syntactic Ambiguity & SVO vs. SOV Word Order]")
    print(f"  * SVO English: '{PAPER_EXAMPLES['order_svo_en']}'")
    print("    NLTK can construct Context-Free Grammars (nltk.CFG) and Chart Parsers,")
    print("    explicitly producing two distinct parse trees for 'I saw the man with a camera'.")
    print(f"  * SOV Amharic: '{PAPER_EXAMPLES['order_sov_am']}' and '{PAPER_EXAMPLES['clitic_am_1']}'")
    print("    NLTK cannot parse Amharic syntax without manually written grammar rules.")
    print("    It treats 'በቤታችን' as 1 monolithic token, hiding root 'ቤት'.")

    print("\n[C. Pragmatic Ambiguity: 'በሩ ክፍት ነው' / 'It is cold in here']")
    print("  * NLTK has no pragmatic or intent modeling; operates strictly at syntax/rules.")

    # --------------------------------------------------------------------------
    # 2. spaCy
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 2: spaCy (Industrial-Strength NLP)")
    print("#"*70)
    print("Role: High-throughput production pipeline (Cython / C-level Doc objects).")
    print("\n[A. Lexical Ambiguity]")
    print("  * English: Disambiguates 'Crane' using pre-trained neural POS and transition-based taggers.")
    print("  * Amharic: 'spacy.blank(\"am\")' only splits text based on punctuation rules.")
    print("    Because spaCy lacks an official pre-trained Amharic model, it CANNOT assign")
    print("    POS tags to differentiate 'በላ' (VERB) from 'በላ' (INTJ), or 'አለ' (VERB vs AUX).")

    print("\n[B. Syntactic Ambiguity & Word Order (SVO vs. SOV)]")
    print("  * English: spaCy's dependency parser identifies PP attachments (e.g. 'with a camera' -> 'saw' vs 'man').")
    print("  * Amharic: Cannot parse Subject-Object-Verb dependency trees for 'ልጁ ምግቡን በላ።'")
    print("    Cannot segment clitic 'በቤታችን' into preposition + noun + suffix.")

    print("\n[C. Pragmatic Ambiguity]")
    print("  * spaCy does not model conversational context or pragmatic speaker intent.")

    # --------------------------------------------------------------------------
    # 3. amseg
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 3: amseg (Amharic Sentence & Word Tokenizer)")
    print("#"*70)
    print("Role: Surface-level orthographic and Ethiopic boundary detection.")
    segmenter = AmsegSegmenter()
    paper_test_text = "ልጁ ምግቡን በላ። በቤታችን ሰላም አለ።"
    sent_res = segmenter.tokenize_sentences(paper_test_text)
    word_res = segmenter.tokenize_words(paper_test_text)
    print(f"Input Text: '{paper_test_text}'")
    print(f"Sentence Segmentation: {sent_res}")
    print(f"Word Tokenization    : {word_res}")
    print("\n[Evaluation against Paper Challenges]:")
    print("  * Strengths: Successfully detects Ethiopic Arat Neteb ('።') and protects abbreviations")
    print("    (e.g., 'ዓ.ም.', 'ት/ቤት') from false sentence splits.")
    print("  * Core Shortcoming (Section II.B & III.B.1): amseg conducts purely surface orthographic")
    print("    segmentation. 'በቤታችን' remains intact as 1 token instead of ['በ', 'ቤት', 'አችን'].")
    print("  * Lexical/Pragmatic: Zero morphological, syntactic, or semantic disambiguation.")

    # --------------------------------------------------------------------------
    # 4. fidel-tools
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 4: fidel-tools (Script Normalization & Cleaning)")
    print("#"*70)
    print("Role: Ethiopic homophone grapheme unification and character standardization.")
    normalizer = FidelToolsNormalizer()
    homophone_sample = "ሐሳብ እና ኀይል በሠላም ተዋሐዱ። ፀሐይ ወጣች።"
    normalized_res = normalizer.normalize(homophone_sample)
    print(f"Raw Fidel Input : {homophone_sample}")
    print(f"Normalized Text : {normalized_res}")
    print("\n[Evaluation against Paper Challenges]:")
    print("  * Resolves Lexical Dilution (Section II.A & III.B.2): Converts 'ሀ/ሐ/ኀ', 'ሰ/ሠ', 'አ/ዐ', 'ጸ/ፀ'")
    print("    into canonical graphemes, preventing vocabulary fragmentation across news/social media.")
    print("  * Crucial Limitation: Over-normalization flattens etymological nuances and proper nouns.")
    print("    It does not perform lemmatization or syntactic parsing.")

    # --------------------------------------------------------------------------
    # 5. amharic-text-processor
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 5: amharic-text-processor (End-to-End Cleaning & Stemming)")
    print("#"*70)
    print("Role: Preprocessing wrapper bundling cleaning, stop-words, and heuristic stemming.")
    proc = AmharicTextProcessor()
    clitic_input = "ልጁ በቤታችን ውስጥ አልሄደም"
    cleaned_stems = proc.clean_and_preprocess(clitic_input)
    print(f"Input Sentence   : '{clitic_input}'")
    print(f"Processed Tokens : {cleaned_stems}")
    print("\n[Evaluation against Paper Challenges]:")
    print("  * Partially tackles Clitic Agglutination (Section II.B): Heuristically strips prefixes ('በ-')")
    print("    and suffixes ('-አችን'), mapping 'በቤታችን' closer to its root 'ቤት'.")
    print("  * Flaw (Section III.B.3 & IV): Rule-based stemmer frequently causes over-stemming on short roots")
    print("    and under-stemming on circumfixed verbs like 'አልሄደም' (negation circumfix አል-...-ም).")

    # --------------------------------------------------------------------------
    # 6. Traditional N-Grams (nltk.lm)
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 6: Traditional N-Grams (nltk.lm)")
    print("#"*70)
    print("Role: Statistical sequence probability estimation via Markov assumption.")
    lm = TraditionalNGramLM(n=2)
    toy_corpus = [
        ["ልጁ", "ምግቡን", "በላ"],
        ["ልጁ", "በመኪና", "ሄደ"],
        ["በቤታችን", "ሰላም", "አለ"]
    ]
    lm.train(toy_corpus)
    prob_sov = lm.probability("በላ", ("ምግቡን",))
    prob_svo = lm.probability("ምግቡን", ("በላ",))
    print(f"Trained on Amharic SOV corpus: {toy_corpus}")
    print(f"P('በላ' | 'ምግቡን') [SOV: Object -> Verb]  = {prob_sov:.4f}")
    print(f"P('ምግቡን' | 'በላ') [SVO: Verb -> Object]  = {prob_svo:.4f}")
    print("\n[Evaluation against Paper Challenges]:")
    print("  * Word Order (Section II.B): Captures local n-gram transition statistics for SOV order.")
    print("  * Failure Modes (Section III.C):")
    print("    1. Severe Sparsity: Cannot handle out-of-vocabulary inflected forms.")
    print("    2. Context Blindness: Cannot resolve long-range dependencies or PP attachments ('ልጁን በመኪና ያዘው').")
    print("    3. Lexical Ambiguity: Assigns identical probability distributions to polysemous words ('በላ').")

    # --------------------------------------------------------------------------
    # 7. Hugging Face Transformers
    # --------------------------------------------------------------------------
    print("\n" + "#"*70)
    print("FRAMEWORK 7: Hugging Face Transformers (mBERT, XLM-RoBERTa, AfroXLMR)")
    print("#"*70)
    print("Role: Deep contextual self-attention representations & subword tokenization.")
    subword_demo = SubwordTokenizerDemo()
    complex_words = ["በቤታችን", "አልሄደም", "ልጁን"]
    print("Subword Tokenization (SentencePiece / WordPiece simulation):")
    for cw in complex_words:
        print(f"  Word: '{cw:<10}' -> Subwords: {subword_demo.tokenize(cw)}")

    print("\n[Evaluation against Paper Challenges]:")
    print("  * Resolves Lexical Ambiguity (Section II.A): Dynamic self-attention generates distinct")
    print("    contextual embeddings for 'በላ' (ate) vs. 'በላ' (speak up) based on full sentence context.")
    print("  * Resolves Syntactic & SOV Divergence (Section II.B): Subword tokenizers break agglutinative")
    print("    words ('በቤታችን' -> 'በ' + '##ቤት' + '##አችን') eliminating OOV errors, while bidirectional")
    print("    attention captures long-distance Subject-Verb relationships.")
    print("  * Addresses Pragmatic Ambiguity (Section II.C): Fine-tuned LLMs can infer indirect requests")
    print("    (e.g., 'በሩ ክፍት ነው' -> request to close the door) through task-specific instruction tuning.")
    print("  * Critical Bottleneck (Section III.C & IV.3): 'Token Fertility' inflation on Ethiopic script.")
    print("    Because Ge'ez is underrepresented in pre-training vocabularies, words are split into")
    print("    excessive fragments, increasing latency and reducing effective context window length.")
    print("="*80 + "\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    evaluate_frameworks_on_paper()
