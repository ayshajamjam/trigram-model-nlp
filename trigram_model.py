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
    return set(word for word in word_counts if word_counts[word] > 1), len(word_counts)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
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

    # print('Modified List: ', modified_list)

    tuple_list = []
    for i in range(0, len(modified_list)):
        single_tuple = modified_list[i:i+n]
        tuple_list.append(tuple(single_tuple))
        if(modified_list[i+(n-1)] == 'STOP'):
            break

    # print('N Tuple List: ', tuple_list)

    return tuple_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon, self.lexicon_size = get_lexicon(generator)
        print(self.lexicon_size)
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
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
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

        # print(self.unigramcounts[('the',)])
        # print(self.bigramcounts[('START','the')])
        # print(self.trigramcounts[('START','START','the')])
        # print('Unigram size: ', unigram_size)
        # print('Bigram size: ', bigram_size)
        # print('Trigram size: ', trigram_size)
        # print('\n')

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        trigram_count = self.trigramcounts[trigram]
        # print('Input trigram count: ', trigram_count)
        
        bigram = tuple((trigram[0:2]))
        # print(tuple((trigram[1:3])))

        if(bigram == (('START', 'START'))):
            bigram_count = self.numberOfSentences
            # print('Bigram count (# sentences): ', bigram, ' ', bigram_count)
        elif(self.bigramcounts[bigram] == 0):
            return (float)(1/self.lexicon_size)
            # print('Bigram count ("unseen"): ', bigram_count)
        else:
            bigram_count = self.bigramcounts[bigram]
            # print('Bigram count: ', bigram, ' ', bigram_count)

        raw_probability = trigram_count/bigram_count

        # print('Raw input trigram probability: ', raw_probability)

        return raw_probability

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        bigram_count = self.bigramcounts[bigram]

        # print(bigram)
        # print('Input bigram count: ', bigram_count)

        unigram = tuple([(bigram[0])])
        # print(unigram)
        if(unigram == ('START',)):
            unigram_count = self.numberOfSentences
            # print('Unigram count (# sentences): ', unigram, ' ', unigram_count)
        else:
            unigram_count = self.unigramcounts[unigram]
            # print('Unigram count: ', unigram, ' ', unigram_count)

        raw_probability = bigram_count/unigram_count
        # print('Raw input bigram probability: ', raw_probability)

        return raw_probability
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        input_count = self.unigramcounts[unigram]

        # if this method has already been calculated before, find answer in TrigramModel instance
        total = 0
        if(self.numberOfWords):
            total = self.numberOfWords
        else:
            # size of the lexicon (unique words) 
            # total = len(self.unigramcounts)
            # # including non-unique unigrams
            for i in self.unigramcounts:
                if(i != ('START',)):
                    total += self.unigramcounts[i]
            self.numberOfWords = total

        raw_probability = input_count/total

        # print('Input unigram count: ', input_count)
        # print('Total number of unigrams: ', total)
        # print('Raw input unigram probability: ', raw_probability)

        return raw_probability

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
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

        # print('Smoothed Trigram probability: ', smoothed_probability)

        return smoothed_probability
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        ngrams = get_ngrams(sentence, 3)
        log_probabilities = 0

        for trigram in ngrams:
            # print(trigram)
            log_probabilities += math.log2(self.smoothed_trigram_probability(trigram))

        # print('Log probability:', log_probabilities)
        return log_probabilities

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        corpus_log_probabilites = 0
        test_number_of_words = 0

        for sentence in corpus:
            # print(sentence)
            test_number_of_words += len(sentence) + 1
            corpus_log_probabilites += self.sentence_logprob(sentence)

        l = (float)(corpus_log_probabilites/test_number_of_words)
        print("l: ", l)
        perplexity = math.pow(2, -l)
        print('Perplexity: ', perplexity)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    #### My tests
    ## Part 1
    # get_ngrams(["natural","language","processing"],1)

    # Part 3
    # model.raw_unigram_probability(('the',))
    print(model.raw_bigram_probability(('START', 'the')))
    print(model.raw_trigram_probability(('START','START', 'the')))

    # print('\n')
    # model.raw_bigram_probability(('the','jurors'))
    # print('\n')
    # model.raw_trigram_probability(('START','START','the'))

    # print(model.raw_unigram_probability(('waiting',)))
    # print(model.raw_bigram_probability(('START', 'waiting')))
    # print(model.raw_trigram_probability(('START', 'START', 'waiting')))
    # print(model.smoothed_trigram_probability(('START', 'START', 'waiting')))

    # Part 4
    # model.smoothed_trigram_probability(('START','START','waiting'))

    # Part 5
    # sentence = ["natural","language","processing"]
    # model.sentence_logprob(sentence)

    # Part 6
   

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
