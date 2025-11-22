import random
import nltk
from nltk.corpus import semcor, wordnet as wn
from wsd import (
    build_dataset,
    simplified_lesk,
    most_frequent_sense_predictor,
    STOPWORDS,
    RANDOM_SEED,
    NUM_SENTENCES,
)

def pretty_syn(s):
    if s is None:
        return "None"
    return f"{s.name()}  ({s.definition()})"

def main():
    dataset = build_dataset(NUM_SENTENCES, RANDOM_SEED)

    samples = []
    for sent_tokens, gold_synsets in dataset:
        for i, (word, gold) in enumerate(zip(sent_tokens, gold_synsets)):
            if gold is None or not wn.synsets(word.lower()):
                continue

            mfs = most_frequent_sense_predictor(sent_tokens, i, word.lower())
            lesk = simplified_lesk(word.lower(), sent_tokens, STOPWORDS)

            if mfs == gold and lesk != gold:
                label = "MFS correct only"
            elif lesk == gold and mfs != gold:
                label = "Lesk correct only"
            elif lesk == gold and mfs == gold:
                label = "Both correct"
            else:
                label = "Both wrong"

            samples.append({
                "label": label,
                "sentence": " ".join(sent_tokens),
                "word": word,
                "gold": pretty_syn(gold),
                "mfs": pretty_syn(mfs),
                "lesk": pretty_syn(lesk)
            })

    random.seed(10)
    random.shuffle(samples)

    chosen = []
    groups = ["Lesk correct only", "MFS correct only", "Both wrong", "Both correct"]
    for g in groups:
        for s in samples:
            if s["label"] == g:
                chosen.append(s)
                break

    for s in chosen:
        print("\n---")
        print("Case:", s["label"])
        print("Sentence:", s["sentence"])
        print("Target word:", s["word"])
        print("Gold:", s["gold"])
        print("MFS:", s["mfs"])
        print("Lesk:", s["lesk"])

if __name__ == "__main__":
    main()
