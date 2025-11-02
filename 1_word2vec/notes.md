# Word2Vec and Word Embeddings

- Notes on language in general
  - It's not just a way to communicate/coordinate between humans (which is something that differentiataes humans from chimps) but it also allows us to form higher order thoughts and gives us a scaffolding for thinking, reasoning, and abstract ideas.
  - We are interested in meaning of words - want to help computers understand the meaning of words
- Earlier approaches like WordNet created lists of synonyms and antonyms
- Idea: representing words as vectors. Naive idea would be sparse one-hot vectors but those have no notion of similarity built in
- Next idea: distributional semantics - a word's meaning is given by the words that frequently appear close to it. The words that appear around a word give information about its meaning.
- So you can use the many contexts in which a word appears to build up a representation of the word.

Word vectors:

- Use short, dense vectors to represents words. These are word vectors or embeddings. The similarity of two words can be calculated by the dot product between two vectors
- We want to learn "good" word vectors - ie ones that correctly capture the "closeness" or "farness" of different words
- In high dimensions words can be close to each other on many different dimensions. Each dimension represents a kind of "aspect" of language or "context" in which a word might appear

Word2Vec

- Idea:
  - We have a large body of text (corpus)
  - Every word is represented by a vector
  - For each position t in the text:
    - calculate the probability of the outside words o1,o2.. occuring adjacent to center word c using the similarity of the vectors v_c and v_o1...
    - Keep adjusting the word vectors to maximize the overall probability over the whole corpus

In other words, we are maximizing the likelihood function

$$L(\theta) = \prod_{t=1}^T \prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j}|w_t; \theta)$$

where $t=1...T$ represent the words in the corpus, m is the size of the sliding window, $w_t$ represents the word vector for word $t$ and $\theta$ represents the combined vector of all the word vectors.

How to calculate $P(w_{t+j}|w_t; \theta)$?

- We will use two word vectors per word $w$ in the corpus:
  - $v_w$ when $w$ is a center word
  - $u_w$ when $w$ is a context word
- Then for center word $c$ and context word $o$:
  $$P(o|c) = \frac{exp(u_o^T v_c)}{\sum_{w \in V}exp(u_w^T v_c)}$$

This function above is called the _softmax_ function $\R^n \rightarrow (0,1)^n$

$$softmax(x)_i = \frac{exp(x_i)}{\sum_{j=1}^n exp(x_j)}$$

In simple words, it turns a vector of real numbers of length n into a probability distribution.

So where to do these word vectors come from and how do we "learn" them from the corpus of data?

TODO:

- How to formulate as a learning problem
- Stochastic gradient descent
- Deriving the gradient by hand
- Skipgram v/s CBOW formulation
- Negative sampling idea
- GloVe model
- Evaluation: intrinsic v/s extrinsic
- Named Entity Recognition and Binary neural classifier

## Resources

[McCormick article](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

[Jake Tae article](https://jaketae.github.io/study/word2vec/)

[Chris Manning course lecture 1](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)
