# Essay Classification with n-Gram Model Natural Language Processing

## About This Project
This is a trigram language model built in Python. The model stores raw counts of n-gram occurrences and then computes the probabilities on demand, allowing for smoothing. I used supervised learning to train the model on brown_train.txt and used brown_test.txt to test the model.

## Run

python3 trigram_model.py brown_train.txt brown_train.txt
python3 trigram_model.py brown_train.txt brown_test.txt

## Part 1 - extracting n-grams from a sentence

The function get_ngrams takes a list of strings and an integer n as input, and returns padded n-grams over the list of strings. The result is a list of Python tuples. 

For example: 

```
>>> get_ngrams(["natural","language","processing"], 1)
[('START',), ('natural',), ('language',), ('processing',), ('STOP',)]

>>> get_ngrams(["natural","language","processing"], 2)
('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]

>>> get_ngrams(["natural","language","processing"], 3)
[('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]
```

## Part 2 - counting n-grams in a corpus

We will work with two different data sets. The first data set is the Brown corpus, which is a sample of American written English collected in the 1950s. The format of the data is a plain text file brown_train.txt, containing one sentence per line. Each sentence has already been tokenized.

**Reading the Corpus and Dealing with Unseen Words**

corpus_reader() takes the name of a text file as a parameter and returns a Python generator object. Generators allow you to iterate over a collection, one item at a time without ever having to represent the entire data set in a data structure (such as a list). This is a form of lazy evaluation. You could use this function as follows: 

```
>>> generator = corpus_reader("")
>>> for sentence in generator:
             print(sentence)

['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of', 'atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']
['the', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'city', 'executive', 'committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'city', 'of', 'atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.']
['the', 'september-october', 'term', 'jury', 'had', 'been', 'charged', 'by', 'fulton', 'superior', 'court', 'judge', 'durwood', 'pye', 'to', 'investigate', 'reports', 'of', 'possible', '``', 'irregularities', "''", 'in', 'the', 'hard-fought', 'primary', 'which', 'was', 'won', 'by', 'mayor-nominate', 'ivan', 'allen', 'jr', '&', '.']
```

Note that iterating over this generator object works only once. After you are done, you need to create a new generator to do it again. 

There are two sources of data sparseness when working with language models: Completely unseen words and unseen contexts. One way to deal with unseen words is to use a pre-defined lexicon before we extract ngrams. The function corpus_reader has an optional parameter lexicon, which should be a Python set containing a list of tokens in the lexicon. All tokens that are not in the lexicon will be replaced with a special "UNK" token.

Instead of pre-defining a lexicon, we collect one from the training corpus. This is the purpose of the function get_lexicon(corpus). This function takes a corpus iterarator (as returned by corpus_reader) as a parameter and returns a set of all words that appear in the corpus more than once. The idea is that words that appear only once are so rare that they are a good stand-in for words that have not been seen at all in unseen text.

When a new TrigramModel is created, we pass in the filename of a corpus file to the TrigramModel constructor. We then iterate through the corpus twice: once to collect the lexicon, and once to count n-grams.

**Counting n-grams**

The method count_ngrams counts the occurrence frequencies for ngrams in the corpus. The method creates three instance variables of TrigramModel, which store the unigram, bigram, and trigram counts in the corpus. Each variable is a dictionary (a hash map) that maps the n-gram to its count in the corpus.

For example, after populating these dictionaries, we want to be able to query

```
>>> model.trigramcounts[('START','START','the')]
5478
>>> model.bigramcounts[('START','the')]
5478
>>> model.unigramcounts[('the',)]
61428
```

Where model is an instance of TrigramModel that has been trained on a corpus. Note that the unigrams are represented as one-element tuples (indicated by the , in the end). 

## Part 3 - Raw n-gram probabilities

The methods raw_trigram_probability(trigram), raw_bigram_probability(bigram), and 
raw_unigram_probability(unigram) return an unsmoothed probability computed from the trigram, bigram, and unigram counts.

## Part 4 - Smoothed probabilities

The method smoothed_trigram_probability(self, trigram) uses linear interpolation between the raw trigram, unigram, and bigram probabilities. 

Interpolation parameters are set to lambda1 = lambda2 = lambda3 = 1/3.

## Part 5 - Computing Sentence Probability

The method sentence_logprob(sentence) returns the log probability of an entire sequence. I use the get_ngrams function to compute trigrams and the smoothed_trigram_probability method to obtain probabilities. I then convert each probability into logspace using math.log2().

Then, instead of multiplying probabilities, I add the log probabilities. Regular probabilities would quickly become too small because of repeated multiplications of numbers between (0,1), leading to numeric issues.

Adding in log space is equivalent to multiplying in linear space.

## Part 6 - Perplexity

perplexity(corpus) computes the perplexity of the model on an entire corpus. 
Corpus is a corpus iterator (as returned by the corpus_reader method). 
Perplexity is defined as 2^-l, where l is defined as: 

```
l = 1/M \sum\limits_{i=1}^m \log p(s_i)
```

Here M is the total number of words. So to compute the perplexity, sum the log probability for each sentence, and then divide by the total number of words tokens in the corpus to normalize.

Runingn the perplexity function on the test set for the Brown corpus brown_test.txt gives a result less than 400.

This is a form of intrinsic evaluation.

## Part 7 - Using the Model for Text Classification

In this final part of the problem I apply the trigram model to a text classification task, using a data set of essays written by non-native speakers of English for the ETS TOEFL test. These essays are scored according to skill level low, medium, or high. We will only consider essays that have been scored as "high" or "low". We will train a different language model on a training set of each category and then use these models to automatically score unseen essays. We compute the perplexity of each language model on each essay. The model with the lower perplexity determines the class of the essay. 

The files ets_toefl_data/train_high.txt and ets_toefl_data/train_low.txt in the data zip file contain the training data for high and low skill essays, respectively. The directories ets_toefl_data/test_high and ets_toefl_data/test_low contain test essays (one per file) of each category. I cannot share this data set since it is proprietary and licensed to Columbia University.

The method essay_scoring_experiment is called by passing two training text files, and two testing directories (containing text files of individual essays). It returns the accuracy of the prediction.

The method creates two trigram models, reads in the test essays from each directory, and computes the perplexity for each essay. I compare the perplexities and return the accuracy (correct predictions / total predictions). 

On the essay data set, I get an accuracy of > 80%.