import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    PART 1- Given a sequence, this function returns a list of n-grams, where each n-gram is a Python tuple.
    This works for arbitrary values of n >= 1 
    """

    if (n < 1):
        print("please input n value >= 1")
        return

    # Adding STARTS and one STOP
    modified_list = ['START']
    if (n > 2):
        for i in range(n-2):
            modified_list += ['START']
    modified_list += sequence + ['STOP']

    tuple_list = []
    for i in range(0, len(modified_list)):
        single_tuple = modified_list[i:i+n]
        tuple_list.append(tuple(single_tuple))
        if(modified_list[i+(n-1)] == 'STOP'):
            break

    return tuple_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.numberOfSentences = 0
        self.numberOfWords = 0
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        PART 2) Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        unigram_size = 0
        bigram_size = 0
        trigram_size = 0

        for sentence in corpus:
            self.numberOfSentences += 1

            unigram_list = get_ngrams(sentence, 1)
            bigram_list = get_ngrams(sentence, 2)
            trigram_list = get_ngrams(sentence, 3)

            for tup in unigram_list:
                if(tup != ('START',)):
                    unigram_size += 1
                    self.unigramcounts[tup] += 1

            for tup in bigram_list:
                bigram_size += 1
                self.bigramcounts[tup] += 1

            for tup in trigram_list:
                trigram_size += 1
                self.trigramcounts[tup] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        PART 3) Returns the raw (unsmoothed) trigram probability
        """

        trigram_count = self.trigramcounts[trigram]
        
        bigram = tuple((trigram[0:2]))

        if(bigram == (('START', 'START'))):
            bigram_count = self.numberOfSentences
        elif(self.bigramcounts[bigram] == 0):
            return (float)(1/(len(self.unigramcounts)))
        else:
            bigram_count = self.bigramcounts[bigram]

        raw_probability = trigram_count/bigram_count

        return raw_probability

    def raw_bigram_probability(self, bigram):
        """
        PART 3) Returns the raw (unsmoothed) bigram probability
        """

        bigram_count = self.bigramcounts[bigram]

        unigram = tuple([(bigram[0])])
        if(unigram == ('START',)):
            unigram_count = self.numberOfSentences
        else:
            unigram_count = self.unigramcounts[unigram]

        raw_probability = bigram_count/unigram_count

        return raw_probability
    
    def raw_unigram_probability(self, unigram):
        """
        PART 3) Returns the raw (unsmoothed) unigram probability.
        """

        input_count = self.unigramcounts[unigram]

        # if this method has already been calculated before, find answer in TrigramModel instance
        total = 0
        if(self.numberOfWords):
            total = self.numberOfWords
        else:
            # including non-unique unigrams
            for i in self.unigramcounts:
                if(i != ('START',)):
                    total += self.unigramcounts[i]
            self.numberOfWords = total

        raw_probability = input_count/total

        return raw_probability    

    def smoothed_trigram_probability(self, trigram):
        """
        PART 4) Returns the smoothed trigram probability (using linear interpolation). 
        """

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        product_1 = lambda1 * self.raw_trigram_probability(trigram)
        bigram = tuple(trigram[1:3])
        product_2 = lambda2 * self.raw_bigram_probability(bigram)
        unigram = tuple(trigram[2:4])
        product_3 = lambda3 * self.raw_unigram_probability(unigram)

        smoothed_probability = product_1 + product_2 + product_3

        return smoothed_probability
        
    def sentence_logprob(self, sentence):
        """
        PART 5) Returns the log probability of an entire sequence.
        """

        ngrams = get_ngrams(sentence, 3)
        log_probabilities = 0

        for trigram in ngrams:
            log_probabilities += math.log2(self.smoothed_trigram_probability(trigram))

        return log_probabilities

    def perplexity(self, corpus):
        """
        PART 6) Returns the log probability of an entire sequence.
        """

        corpus_log_probabilites = 0
        test_number_of_words = 0

        for sentence in corpus:
            test_number_of_words += len(sentence) + 1
            corpus_log_probabilites += self.sentence_logprob(sentence)

        l = (float)(corpus_log_probabilites/test_number_of_words)
        perplexity = math.pow(2, -l)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)   # model trained on high-scoring essays
        model2 = TrigramModel(training_file2)   # model trained on low-scoring essays

        total = 0
        correct = 0

        # Testing high scores
        for f in os.listdir(testdir1):
            pp_mod1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_mod2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if(pp_mod1 < pp_mod2):
                correct += 1
            total += 1
    
        # Testing low scores
        for f in os.listdir(testdir2):
            pp_mod1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_mod2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if(pp_mod2 < pp_mod1):
                correct += 1
            total += 1 
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print('Perplexity: ', pp)

    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    print('Accuracy: ', acc)
    
