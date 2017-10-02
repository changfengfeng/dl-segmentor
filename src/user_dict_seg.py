# -*- coding: utf-8 -*-

import numpy as np
import queue

class TrieNode:
    """ TrieNode for the use define dict

    Root is the entry point of the dict
    node.children {char, node_idx} map
    node.fail_node: when find failed，next match node in the tree：
              a b c d
              b c
        when failed at abc，it will search from bc
    """
    def __init__(self):
        self.leaf = False
        self.fail_node = None
        self.len = 0
        self.weight = 0
        self.children = {}

class TrieTree:

    def __init__(self):
        self.root = TrieNode()
        self.root.fail_node = self.root
        self.num_node = 1

    def push_node(self, word, weight):
        """ add node to the trie tree

        Args:
            word: the user define word
            weigth: the weight (>0) of the word
        """
        cur = self.root
        prev = None
        if len(word) == 0:
            return
        for i in range(len(word)):
            char = word[i]
            prev = cur
            if char in cur.children:
                cur = cur.children[char]
            else:
                new_node = TrieNode()
                cur = new_node
                prev.children[char] = cur
                self.num_node += 1

        if cur.leaf:
            return
        else:
            cur.leaf = True
            cur.weight = weight
            cur.len = len(word)

    def build_fail_node(self):
        todos = queue.Queue()
        for k, v in self.root.children.items():
            v.fail_node = self.root
            todos.put(v)

        while not todos.empty():
            parent = todos.get()
            for k, v in parent.children.items():
                wc = k
                cur = v
                parent_fail_node = parent.fail_node
                it2 = parent_fail_node.children[wc] if wc in parent_fail_node.children else None
                while parent_fail_node != self.root and it2 == None:
                    parent_fail_node = parent_fail_node.fail_node
                    it2 = parent_fail_node.children[wc] if wc in parent_fail_node.children else None
                if it2 == None:
                    cur.fail_node = self.root
                else:
                    cur.fail_node = it2
                todos.put(cur)

    def read_from_file(self, dict_fn):
        with open(dict_fn, "r") as f:
            for line in f:
                word, weight = line.split(" ")
                weight = int(weight)
                self.push_node(word, weight)
        if self.num_node > 1:
            self.build_fail_node()

class UserScore:
    """ For echo sentece output the user define dict score for every tags
    """

    def __init__(self, sentence, trie_tree):
        self.sentence = sentence
        self.max_seq_length = len(sentence)
        self.trie_tree = trie_tree
        #[;,0] for the flag if it been found in the user dict
        self.weights = np.ones([self.max_seq_length, 5], "int32")
        self.weights[:,0] = 0
        self.scores = np.zeros([self.max_seq_length, 4], "float32")

    def get_score(self):
        """ scan the sentence, stats the every char tag prob using trie tree
        """
        self.scan()
        for i in range(len(self.weights)):
            if self.weights[i,0] == 1:
                total = sum(self.weights[i,1:])
                self.scores[i] = np.log(self.weights[i,1:] / total)
        return self.scores

    def report_word(self, pos, weight, word_len):
        #print(pos, weight, word_len)
        if word_len == 1:
            self.weights[pos, 0] = 1
            self.weights[pos, 1] += weight
        else:
            for i in range(word_len):
                p = pos - word_len + 1 + i
                self.weights[p, 0] = 1
                if i == 0:
                    self.weights[p, 2] += weight
                elif i == word_len - 1:
                    self.weights[p, 4] += weight
                else:
                    self.weights[p, 3] += weight
        #print(self.weights)

    def scan(self):
        """ scan the sentence using trie tree
        """
        if len(self.sentence) == 0:
            return
        cur = self.trie_tree.root
        parent = None
        prev_found = None
        prev_pos = 0
        i = 0
        while i < len(self.sentence):
            wc = self.sentence[i]
            parent = cur
            if wc in cur.children:
                cur = cur.children[wc]
            else:
                cur = None

            if cur == None:
                if prev_found != None:
                    self.report_word(prev_pos, prev_found.weight,
                            prev_found.len)
                    i = prev_pos - prev_found.len + 2
                    prev_found = None
                    cur = self.trie_tree.root
                    continue
                else:
                    cur = parent.fail_node
                    if parent != self.trie_tree.root:
                       i = i - 1
            if cur.leaf:
                prev_found = cur
                prev_pos = i
            i = i + 1
        if prev_found != None:
            self.report_word(prev_pos, prev_found.weight, prev_found.len)
"""
if __name__ == "__main__":
    trie_tree = TrieTree()
    trie_tree.read_from_file("model/user_dict.txt")
    user_scorer = UserScore("挑战中共国际创辉煌国际", trie_tree)
    score = user_scorer.get_score()
    print(score)
"""

