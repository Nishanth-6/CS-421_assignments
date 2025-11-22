import nltk
from nltk.corpus import semcor, wordnet as wn
from nltk.metrics import recall, precision, f_measure

nltk.download('wordnet', quiet=True)
nltk.download('semcor', quiet=True)


def load_sentences(num):
    """
    Load and return the first num sentences from SemCor.
    """
    try:
        return semcor.sents()[:num]
    except LookupError:
        nltk.download('semcor')
        return semcor.sents()[:num]


def load_tagged_sents(num):
    """
    Load and return the first num tagged sentences from SemCor.
    """
    try:
        return semcor.tagged_sents(tag='sem')[:num]
    except LookupError:
        nltk.download('semcor')
        return semcor.tagged_sents(tag='sem')[:num]


# DO NOT MODIFY
def process_labels(sentences):
    sents = []
    labels = []
    for sent in sentences:
        curr_sent = []
        curr_labels = []
        for word in sent:
            if isinstance(word, nltk.Tree):
                lemma = word.label()
                text = "_".join(word.leaves())
                try:
                    if 'group.n.' not in lemma.synset().name():
                        curr_sent.append(text)
                        curr_labels.append(lemma.synset().name())
                except:
                    curr_sent.append(text)
                    curr_labels.append(lemma)
        sents.append(curr_sent)
        labels.append(curr_labels)
    return sents, labels


def most_freq_sense_model(word):
    """
    Return WordNet's most frequent sense (1st synset name as string) or None.
    """
    synsets = wn.synsets(word)
    return synsets[0].name() if synsets else None


def get_most_freq_predictions(sentences):
    """
    Apply MFS model to each word in each sentence.
    Return list of list of synset names (strings or None).
    """
    preds = []
    for sent in sentences:
        curr = []
        for word in sent:
            curr.append(most_freq_sense_model(word.lower()))
        preds.append(curr)
    return preds


def lesk_model(word, sentence):
    """
    Implement simplified Lesk algorithm.
    Return synset name (string) or None.
    """
    context = set(w.lower() for w in sentence)
    best_sense = None
    max_overlap = 0

    for sense in wn.synsets(word):
        # Get definition and examples
        signature = set()
        signature.update(sense.definition().lower().split())
        for example in sense.examples():
            signature.update(example.lower().split())

        overlap = len(signature.intersection(context))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense.name()

    return best_sense


def get_lesk_predictions(sentences):
    """
    Run Lesk model on all tokens.
    Return list of list of synset name strings or None.
    """
    preds = []
    for sent in sentences:
        curr = []
        for word in sent:
            curr.append(lesk_model(word.lower(), sent))
        preds.append(curr)
    return preds


def evaluate(labels, predicted):
    """
    Evaluate using NLTK precision, recall, and f-measure.
    Flat list comparison.
    """
    gold_flat = [g for sent in labels for g in sent]
    pred_flat = [p for sent in predicted for p in sent]

    p = precision(set(enumerate(gold_flat)), set(enumerate(pred_flat)))
    r = recall(set(enumerate(gold_flat)), set(enumerate(pred_flat)))
    f = f_measure(set(enumerate(gold_flat)), set(enumerate(pred_flat)))

    return p, r, f


def main():
    sents = load_sentences(50)
    tagged_sents = load_tagged_sents(50)
    processed_sentences, labels = process_labels(tagged_sents)
    print("Sentence 1 words:", processed_sentences[0])
    print("Sentence 1 labels:", labels[0])


    preds_mfs = get_most_freq_predictions(processed_sentences)
    print("MFS:", evaluate(labels, preds_mfs))

    preds_lesk = get_lesk_predictions(processed_sentences)
    print("LESK:", evaluate(labels, preds_lesk))
    
    print("MFS sentence 1:", preds_mfs[0])
    print("LESK sentence 1:", preds_lesk[0])

if __name__ == '__main__':
    main()
