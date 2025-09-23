# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 0
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import json

# Function to read a file
# filepath: full path to the file
# Returns: a string containing file contents
def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read()

# Function to convert a string to a list of lowercase words
# in_str: a string containing the text
# Returns: a list containing lowercase words
def convert_to_words(in_str):
     return in_str.lower().split()


# Function to count the words
# words: a list containing words
# Returns: a dictionary where keys are words and corresponding values are counts
def get_wc(words):
    counts = {}
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    return counts

# Function to store the dictionary as JSON
# dictionary: a python dictionary
# out_filepath: path to output file to store the JSON data
# Returns: None
def to_json(dictionary, out_filepath):
    json_str = json.dumps(dictionary)
    with open(out_filepath, "w", encoding="utf-8") as f:
        f.write(json_str)



# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python text_analysis.py
# It should produce the following output (with correct solution):
#       $ python text_analysis.py
#       File loaded: CHAPTER I.
#       Down the Rabbit-Hol...
#       Words: ['chapter', 'i.', 'down', 'the', 'rabbit-hole']
#       The word alice appeared 221 times.

def main():
    # Read the entire file in a string
    content = read_file('alice.txt')
    # Print first 30 characters
    print(f'File loaded: {content[:30]}...')

    # Obtain words from the content
    words = convert_to_words(content)
    # Print the first 5 words
    print(f'Words: {words[:5]}')

    # Count the words
    word_counts = get_wc(words)
    # Print word counts for alice
    word = 'alice'
    if word in word_counts:
        print(f'The word {word} appeared {word_counts[word]} times.')
    else:
        print("Word not found")

    # Save these counts as a JSON file
    to_json(word_counts, 'word_counts.json')


################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
