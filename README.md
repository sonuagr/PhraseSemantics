# PhraseSemantics
Here we try to get the sematics of phrase using Recursive Neural Networks on the word vectors.
The paper we refer is "Semantic Compositionality Through Recursive Matrix-Vector Spaces i.e. Socher et al., 2012b"
by Richard Socher who has done this work during work his PhD thesis.

Important links:
* Socher's home page: http://www.socher.org/  
* Socher's thesis: http://nlp.stanford.edu/~socherr/thesis.pdf
* Socher's paper: http://www.socher.org/index.php/Main/SemanticCompositionalityThroughRecursiveMatrix-VectorSpaces

``Common problems in other methods and their solutions:``
* Represent text in terms of unordered list of words. Sentiments depend not just on the word meanings but how they are ordered.

In Recursive NN ordering matters. The syntactic rules of natural language are known to be recursive (example sentence??). It being recursive, the input need not be fixed. Matrix is used for each word which captures how it changes the meaning of neighboring words. This matrix is then learned for phrases containing the words and smallers phrases.
* Consider only fixed number of neighbours around each word.

In NN, effect of a word isn't limited to a fixed no. of words in neighborhood.
* For good results, a lot of manually designed features like relationship with other words, the part of speech, etc are needed.

But NN: Automatically learns features from raw input.

So, Recursive NN not only predict an underlying hierarchical structure but also learn how words compose the meaning of longer phrases inside such structures

Recursive neural tensor network (Socher et al., 2013d) allows both additive and mediated multiplicative interactions between 
vectors and is able to learn several important compositional sentiment effects in language such as
negation and its scope and contrastive conjunctions like but. (?? We aren't using tensor. is it learnt by our model as well??)

How is pretraining done?

Using git:
```
1. Pull from remote: git pull origin master
2. Check status: git status
3. Add files: git add .
4. Commit changes: git commit -m "What changes you made"
5. Push it to remote: git push origin master
6. If some files are deleted, before step 3 do: git add -u .
```
