# Addis Ababa Science and Technology University
# School of post graduate studies
# Department of AI AND Robotics
# Course: NLP
# Course code: ARIN6013
# Submitted to: Dr, Girma N.
# Submission Date: 11/19/2026
# Assignment on: NLP Tool & Framework Review - Practical Code Demonstrations


# ==============================================================================
# 1. NLTK (pip install nltk)
# ==============================================================================
import nltk

text = "ልጁ ምግቡን በላ። በቤታችን ሰላም አለ።"
tokens = nltk.regexp_tokenize(text, pattern=r'[\w]+|[።፣፤፥፡]')
print("NLTK:", tokens)
# -> ['ልጁ', 'ምግቡን', 'በላ', '።', 'በቤታችን', 'ሰላም', 'አለ', '።']


# ==============================================================================
# 2. spaCy (pip install spacy)
# ==============================================================================
import spacy

nlp = spacy.blank("am")
doc = nlp("ልጁ ምግቡን በላ። በቤታችን ሰላም አለ።")
print("spaCy:", [token.text for token in doc])
# -> ['ልጁ', 'ምግቡን', 'በላ', '።', 'በቤታችን', 'ሰላም', 'አለ', '።']


# ==============================================================================
# 3. amseg (pip install amseg)
# ==============================================================================
from amseg.amharicSegmenter import AmharicSegmenter

segmenter = AmharicSegmenter(sent_punct=[], word_punct=[])
text = "ልጁ ምግቡን በላ። በቤታችን ሰላም አለ።"
print("amseg:", segmenter.amharic_tokenizer(text))
# -> ['ልጁ', 'ምግቡን', 'በላ', '።', 'በቤታችን', 'ሰላም', 'አለ', '።']


# ==============================================================================
# 4. fidel-tools (pip install fidel-tools)
# ==============================================================================
from fidel_tools import normalize

text = "ሐሳብ እና ኀይል በሠላም ተዋሐዱ። ፀሐይ ወጣች።"
print("fidel-tools:", normalize(text))
# -> 'ሀሳብ እና ሀይል በሰላም ተዋሀዱ። ጸሀይ ወጣች።'


# ==============================================================================
# 5. amharic-text-processor (pip install amharic-text-processor)
# ==============================================================================
from amharic_text_processor import AmharicTextProcessor

processor = AmharicTextProcessor()
text = "ልጁ በቤታችን ውስጥ አልሄደም! ይህ በጣም አሳዛኝ ነው።"
print("amharic-text-processor:", processor.preprocess(text))
# -> ['ልጁ', 'ቤታችን', 'ሄደም']


# ==============================================================================
# 6. Traditional N-Grams using NLTK (pip install nltk)
# ==============================================================================
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

corpus = [["ልጁ", "ምግቡን", "በላ"], ["በቤታችን", "ሰላም", "አለ"]]
train_data, padded_sents = padded_everygram_pipeline(2, corpus)
model = Laplace(2)
model.fit(train_data, padded_sents)
print("N-Gram P(በላ|ምግቡን):", model.score("በላ", ["ምግቡን"]))
# -> 0.2


# ==============================================================================
# 7. Hugging Face Transformers (pip install transformers torch)
# ==============================================================================
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text = "ልጁ በቤታችን ውስጥ አልሄደም"
print("Hugging Face Subwords:", tokenizer.tokenize(text))
# -> ['ልጁ', 'በ', '##ቤ', '##ታ', '##ች', '##ን', 'ውስጥ', 'አል', '##ሄ', '##ደ', '##ም']
