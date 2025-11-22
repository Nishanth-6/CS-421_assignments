import random
import re
from typing import List, Tuple, Optional

import nltk
from nltk.corpus import semcor, wordnet as wn
from nltk.tree import Tree

# If your template already defines globals, you can wire these in there.
RANDOM_SEED = 42
NUM_SENTENCES = 50

# Simple built in English stopword list so we do not depend on NLTK's stopwords corpus
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "to", "from", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very",
    "can", "will", "just", "do", "does", "did",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "them",
    "my", "your", "his", "their", "our",
    "this", "that", "these", "those"
}


def flatten_semcor_sentence(tagged_sent) -> Tuple[List[str], List[Optional[wn.synset]]]:
    """
    Convert one semcor tagged sentence (tag='sem') into:
      - words: flat list of tokens
      - gold_synsets: list of same length, with a WordNet Synset or None
    """
    words: List[str] = []
    gold_synsets: List[Optional[wn.synset]] = []

    for elem in tagged_sent:
        # elem is either a Tree (with Lemma label) or a list of strings
        if isinstance(elem, Tree):
            label = elem.label()
            synset = None
            # label is usually a WordNet Lemma; we can get its synset
            try:
                synset = label.synset()
            except AttributeError:
                synset = None

            tokens = elem.leaves()  # all surface tokens for this chunk
        else:
            # plain untagged chunk, like ['The']
            synset = None
            tokens = elem

        for tok in tokens:
            words.append(tok)
            gold_synsets.append(synset)

    return words, gold_synsets


def build_dataset(
    num_sentences: int = NUM_SENTENCES,
    seed: int = RANDOM_SEED,
) -> List[Tuple[List[str], List[Optional[wn.synset]]]]:
    """
    Randomly select num_sentences from SemCor and return a list of:
      (sentence_tokens, gold_synsets_per_token)

    sentence_tokens are the raw tokens from semcor.sents().
    gold_synsets_per_token align with those tokens and can be a Synset or None.
    """
    sents = semcor.sents()
    tagged_sents = semcor.tagged_sents(tag="sem")

    assert len(sents) == len(tagged_sents), "SemCor sents and tagged_sents are misaligned"

    random.seed(seed)
    indices = random.sample(range(len(sents)), num_sentences)

    dataset = []
    for idx in indices:
        # raw sentence tokens (as in context)
        raw_tokens = list(sents[idx])
        tagged_sent = tagged_sents[idx]

        flat_tokens, gold_synsets = flatten_semcor_sentence(tagged_sent)

        # In theory, flat_tokens should match raw_tokens; if not, we trust flat_tokens.
        if len(flat_tokens) != len(raw_tokens):
            # Fall back to the flattened tokens for both context and alignment
            sentence_tokens = flat_tokens
        else:
            sentence_tokens = raw_tokens

        dataset.append((sentence_tokens, gold_synsets))

    return dataset


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenizer for WordNet definitions and examples.
    Lowercases, strips non alphabetic boundaries.
    """
    # Split on non alphabetic characters
    tokens = re.split(r"[^a-zA-Z]+", text.lower())
    return [t for t in tokens if t]


def simplified_lesk(
    word: str,
    sentence_tokens: List[str],
    stopword_set: Optional[set] = None,
) -> Optional[wn.synset]:
    """
    Simplified Lesk algorithm for a single word in a given sentence.

    word: target word (string)
    sentence_tokens: full sentence tokens as context (strings)
    stopword_set: set of stopwords to exclude from overlap

    Returns: best sense (Synset) or None if there are no candidate senses.
    """
    if stopword_set is None:
        stopword_set = set()

    # Candidate senses from WordNet
    senses = wn.synsets(word)
    if not senses:
        return None

    # Step 2: best-sense starts as most frequent sense (first synset)
    best_sense = senses[0]
    max_overlap = 0

    # Step 4: context is set of non stopword tokens in the sentence
    context = {
        tok.lower()
        for tok in sentence_tokens
        if tok.lower() not in stopword_set and tok.isalpha()
    }

    for sense in senses:
        signature = set()

        # Definition
        definition_tokens = tokenize_text(sense.definition())
        signature.update(definition_tokens)

        # Examples
        for ex in sense.examples():
            signature.update(tokenize_text(ex))

        # Remove stopwords and non alphabetic
        signature = {
            w for w in signature if w not in stopword_set and w.isalpha()
        }

        # Overlap = size of intersection
        overlap = len(context.intersection(signature))

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


def evaluate_model_on_dataset(
    dataset: List[Tuple[List[str], List[Optional[wn.synset]]]],
    predictor,
) -> Tuple[float, float, float, int, int]:
    """
    Generic evaluation helper.

    dataset: list of (sentence_tokens, gold_synsets_per_token)
    predictor: function(sentence_tokens, index, word_string) -> Synset or None

    Returns:
      precision, recall, f1, num_gold, num_correct
    """
    gold_labels: List[wn.synset] = []
    pred_labels: List[wn.synset] = []

    for sentence_tokens, gold_synsets in dataset:
        for i, (tok, gold_syn) in enumerate(zip(sentence_tokens, gold_synsets)):
            # Only evaluate where SemCor gives a sense and WordNet has some synsets
            if gold_syn is None:
                continue

            # Using lowercased form aligns with typical WordNet usage
            word = tok.lower()
            if not wn.synsets(word):
                continue

            pred_syn = predictor(sentence_tokens, i, word)
            if pred_syn is None:
                # Model could not predict; treat this as no prediction
                continue

            gold_labels.append(gold_syn)
            pred_labels.append(pred_syn)

    num_gold = len(gold_labels)
    if num_gold == 0:
        return 0.0, 0.0, 0.0, 0, 0

    num_correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)

    precision = num_correct / len(pred_labels) if pred_labels else 0.0
    recall = num_correct / num_gold if num_gold else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, num_gold, num_correct


def most_frequent_sense_predictor(sentence_tokens, index, word):
    """
    Predictor for the Most Frequent Sense model.

    Ignores sentence context and always returns the first synset for the given word.
    """
    synsets = wn.synsets(word)
    if not synsets:
        return None
    return synsets[0]


def lesk_predictor(sentence_tokens, index, word):
    """
    Predictor wrapper around simplified_lesk.
    Uses full sentence as context.
    """
    return simplified_lesk(word, sentence_tokens, STOPWORDS)


def print_metrics(name: str, precision: float, recall: float, f1: float, n: int, correct: int):
    print(f"{name} results on SemCor subset ({n} evaluated tokens):")
    print(f"  Correct:   {correct}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 score:  {f1:.4f}")
    print()


def main():
    # Build dataset once, then use it for both models
    print("Building dataset from SemCor...")
    dataset = build_dataset(num_sentences=NUM_SENTENCES, seed=RANDOM_SEED)
    print(f"Selected {len(dataset)} sentences from SemCor.\n")

    # Model 1: Most Frequent Sense
    mfs_p, mfs_r, mfs_f1, mfs_n, mfs_correct = evaluate_model_on_dataset(
        dataset, most_frequent_sense_predictor
    )
    print_metrics("Most Frequent Sense", mfs_p, mfs_r, mfs_f1, mfs_n, mfs_correct)

    # Model 2: Simplified Lesk
    lesk_p, lesk_r, lesk_f1, lesk_n, lesk_correct = evaluate_model_on_dataset(
        dataset, lesk_predictor
    )
    print_metrics("Simplified Lesk", lesk_p, lesk_r, lesk_f1, lesk_n, lesk_correct)


if __name__ == "__main__":
    main()
