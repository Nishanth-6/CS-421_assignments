# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
#
# This code will be graded automatically using Gradescope.
# =========================================================================================================
import nltk
from nltk.corpus import treebank
import numpy as np
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Function: get_treebank_data
# Input: None
# Returns: Tuple (train_sents, test_sents)
#
# This function fetches tagged sentences from the NLTK Treebank corpus, calculates an index for an 80-20 train-test split,
# then splits the data into training and testing sets accordingly.

def get_treebank_data():
    sentences = treebank.tagged_sents()
    split = int(0.8 * len(sentences))
    train_sents = sentences[:split]
    test_sents  = sentences[split:]
    return train_sents, test_sents

# Function: compute_tag_trans_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary A of tag transition probabilities
#
# Iterates over training data to compute the probability of tag bigrams (transitions from one tag to another).

def compute_tag_trans_probs(train_data):
    tag_bigrams = {}
    tag_counts = {}

    for sent in train_data:
        tags = [t for _, t in sent]
        for i in range(1, len(tags)):
            prev, cur = tags[i-1], tags[i]
            tag_bigrams.setdefault(prev, {})
            tag_bigrams[prev][cur] = tag_bigrams[prev].get(cur, 0) + 1
            tag_counts[prev] = tag_counts.get(prev, 0) + 1

    A = {}
    for prev, cur_dict in tag_bigrams.items():
        total = float(tag_counts[prev])
        A[prev] = {cur: cnt / total for cur, cnt in cur_dict.items()}
    return A

# Function: compute_emission_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary B of tag-to-word emission probabilities
#
# Iterates through each sentence in the training data to count occurrences of each tag emitting a specific word, then calculates probabilities.

def compute_emission_probs(train_data):
    emission_counts = {}
    tag_counts = {}

    for sent in train_data:
        for word, tag in sent:
            emission_counts.setdefault(tag, {})
            emission_counts[tag][word] = emission_counts[tag].get(word, 0) + 1
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    B = {}
    for tag, w_counts in emission_counts.items():
        total = float(tag_counts[tag])
        B[tag] = {w: cnt / total for w, cnt in w_counts.items()}
    return B
# Function: viterbi_algorithm
# Input: words (list of words that have to be tagged), A (transition probabilities), B (emission probabilities)
# Returns: List (the most likely sequence of tags for the input words)
#
# Implements the Viterbi algorithm to determine the most likely tag path for a given sequence of words, using given transition and emission probabilities.

def viterbi_algorithm(words, A, B):
    if not words:
        return []

    states = list(B.keys())
    small = 1e-4  # smoothing for unseen transitions/emissions

    # t = 0 initialization (use only emission for the first word, as per template style)
    Vit = [{s: B.get(s, {}).get(words[0], small) for s in states}]
    path = {s: [s] for s in states}

    # recursion
    for t in range(1, len(words)):
        Vit.append({})
        new_path = {}
        w = words[t]
        for cur in states:
            emit = B.get(cur, {}).get(w, small)
            best_prob, best_prev = 0.0, None
            for prev in states:
                trans = A.get(prev, {}).get(cur, small)
                prob = Vit[t-1][prev] * trans * emit
                if prob > best_prob:
                    best_prob, best_prev = prob, prev
            Vit[t][cur] = best_prob
            # best_prev can’t be None because states is non-empty; but guard anyway
            new_path[cur] = (path.get(best_prev, [cur]) + [cur]) if best_prev else [cur]
        path = new_path

    # termination
    last_probs = Vit[-1]
    best_state = max(last_probs, key=last_probs.get)
    return path[best_state]


# Function: evaluate_pos_tagger
# Input: test_data (tagged sentences for testing), A (transition probabilities), B (emission probabilities)
# Returns: Float (accuracy of the POS tagger on the test data)
#
# Evaluates the POS tagger's accuracy on a test set by comparing predicted tags to actual tags and calculating the percentage of correct predictions.

def evaluate_pos_tagger(test_data, A, B):
    correct = 0
    total = 0
    for sent in test_data:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = viterbi_algorithm(words, A, B)
        if not pred or len(pred) != len(gold):
            continue
        for g, p in zip(gold, pred):
            total += 1
            if g == p:
                correct += 1
    return (correct / total) if total else 0.0

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    # Main function to train and evaluate the POS tagger.
    

    train_data, test_data = get_treebank_data()
    A = compute_tag_trans_probs(train_data)
    B = compute_emission_probs(train_data)

    print(f"P(VB -> DT): {A.get('VB', {}).get('DT', 0):.4f}")
    print(f"P(DT -> 'the'): {B.get('DT', {}).get('the', 0):.4f}")
    accuracy = evaluate_pos_tagger(test_data, A, B)
    print(f"Accuracy of the HMM-based POS Tagger: {accuracy:.4f}")
