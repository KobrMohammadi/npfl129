#!/usr/bin/env python3
import os
import sys
import urllib.request


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

    def search(self, word):
        word = word.lower()
        if word in self.variants:
            return self.variants[word]
        else:
            return None

    def add_word(self, word, variants):
        word = word.lower()
        if word not in self.variants:
            self.variants[word] = variants
            print(f"Word '{word}' added to the dictionary.")
        else:
            print(f"Word '{word}' already exists in the dictionary.")

    def remove_word(self, word):
        word = word.lower()
        if word in self.variants:
            del self.variants[word]
            print(f"Word '{word}' removed from the dictionary.")
        else:
            print(f"Word '{word}' not found in the dictionary.")
    def display(self):
        for word, variants in self.variants.items():
            print(f"{word}: {', '.join(variants)}")
    def save(self, filename="updated_dictionary.txt"):
        with open(filename, "w", encoding="utf-8-sig") as f:
            for word, variants in self.variants.items():
                f.write(f"{word} {' '.join(variants)}\n")
        print(f"Dictionary saved as {filename}")

