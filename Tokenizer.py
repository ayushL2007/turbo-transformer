import numpy as np
import pandas as pd
import re

pretokenizer_split_expr = """[\s]{1,}|\s[^A-Za-z0-9]|(?=-)"""


class BPE():
    def __init__(self, vocabSize :int=1000):
        self._nVocab = vocabSize
        self.ctr = 256

    #@params:= sentence :str = should be the raw string without any encoding 
    def pretokenize(self, sentence :str) -> dict[int, tuple[int,...]]:
        Vocab :dict[tuple[int,...], int] = {}
        for i in range(256):
            Vocab[(i)] = i
        words = re.split(pretokenizer_split_expr, sentence)
        i = self.ctr
        for word in words:
            word = tuple(word.encode('utf-8'))
            if word not in Vocab.values():
                Vocab[word] = i
                i+=1

        self.ctr = i
        return Vocab

    #@params:= chunk :tuple[int, ...] = only send after raw ---> raw.encode() ---> list(raw) of type int 
    #param is locally cached from class memory
    def pairwiseRank(self) -> dict[tuple[int, int], int]:
        chunk :tuple[int,...] = self.sentence
        rank :dict[tuple[int, ...], int] = {}
        
        prev = chunk[0]
        for byte in chunk[1:]:
            pair = (prev, byte)
            if pair in rank.keys():
                rank[pair] += 1
            else:
                rank[pair] = 1

            prev = byte
        return rank

    #@params:= sentence :tuple[int, ...] = only send after raw ---> raw.encode() ---> list(raw) of type int 
    def merge(self, pair :tuple[int,int]):
        ctr = self.ctr
        if pair in self._vocab.keys():
            ctr = self._vocab[pair]
        else:
            self._vocab[pair] = ctr
            self.ctr += 1
            
        sentence :tuple[int,...]= self.sentence #local-cache for repeated call
        
        prev = sentence[0]

        #pre-allocation+slicing instead of append to improve runtime against temporary increased ram consumption
        #removes overhead reallocation of append after every k-calls
        newSentence :list[int] = [None]*len(sentence)
        _nNew = 0

        newSentence[_nNew] = prev
        _nNew+=1
        
        for i in range(1, len(sentence)):
            curr = sentence[i]
            p = (prev, curr)
            if p==pair:
                # print(f"merging {pair} into {ctr}")
                newSentence[_nNew-1] = ctr
            else:
                newSentence[_nNew] = curr
                _nNew+=1

            prev = curr

        self.sentence = newSentence[:_nNew]

    def findHighestRank(self, ranks :dict[tuple[int, int], int]) -> tuple[int, int]:
        mostCommonPair :tuple[int, int]= ()
        highestFreq = 0
        for pair, freq in ranks.items():
            if highestFreq < freq:
                mostCommonPair = pair
                highestFreq = freq

        return mostCommonPair

    def preetyPrint(self):
        sentence :list[int]= self.sentence
        Vocab = self._vocab
        for val in sentence:
            if val<256:
                print(chr(val),end="")
            else:
                print("'",end="")
                self.printVocab(val)
                print("'")

    def printVocab(self, key:int):
        if key<256:
            print(chr(key), end="")
            return
        Vocab = self._vocab
        for pair, val in Vocab.items():
            if key==val:
                for subpair in pair:
                    self.printVocab(subpair)
            
                
    def tokenize(self, sentence :str):
        self._vocab = self.pretokenize(sentence)
        
        sentence = sentence.encode('utf-8')
        sentence :list[int] = list(sentence)
        self.sentence = sentence

        singleFreqCount= 0
        for i in range(1000):
            ranks = self.pairwiseRank()
            commonPair = self.findHighestRank(ranks)
            if ranks[commonPair]<=1:
                if singleFreqCount>50:
                    break
                singleFreqCount += 1
            self.merge(commonPair)


a=""" Once upon a time, there was a little girl named Lily. Lily liked to pretend she was a popular princess. She lived in a big castle with her best friends, a cat and a dog. One day, while playing in the castle, Lily found a big cobweb. The cobweb was in the way of her fun game. She wanted to get rid of it, but she was sc...
Once upon a time, in a big lake, there was a brown kayak. The brown kayak liked to roll in the water all day long. It was very happy when it could roll and splash in the lake. One day, a little boy named Tim came to play with the brown kayak. Tim and the brown kayak rolled in the water together. They laughed and had a...
Once upon a time, in a small town, there was a troubled little girl named Lily. She was always sad because she lost her favorite toy, a triangle. She looked everywhere in her house but could not find it. One sunny day, Lily went to the park to play. She saw a big puddle of water and thought her triangle might be there...
Once upon a time, in a peaceful town, there lived a little boy named Tim. Tim loved to run and play outside. One day, Tim saw a race in the park. He was excited and wanted to join the race. Tim went to his friend, Sarah, and said, "Let's start the race!" Sarah smiled and said, "Yes, let's go!" They lined up with the o...
"""
bpe = BPE()
bpe.tokenize(a)
bpe.preetyPrint()