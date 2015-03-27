# PhraseSemantics
Here we try to get the sematics of phrase using Recursive Neural Networks on the word vectors.
The paper we refer is Semantic Compositionality through Recursive Matrix-Vector Spaces by Richard Socher who has done this work during work his PhD thesis.

Important links:
* Socher's home page: http://www.socher.org/  
* Socher's thesis: http://nlp.stanford.edu/~socherr/thesis.pdf

``Common problems in other methods and their solutions:``
* Represent text in terms of unordered list of words. Sentiments depend not just on the word meanings but how they are ordered.

In Recursive NN ordering matters, and we use matrix for each word which captures how it changes the meaning of neighboring words. This matrix is then learned for phrases containing the words and smallers phrases. It being recursive, the input need not be fixed.
* Consider only fixed number of neighbours around each word.

In NN, effect of a word isn't limited to a fixed no. of words in neighborhood.
* For good results, a lot of manually designed features like relationship with other words, the part of speech, etc are needed.

But NN: Automatically learns features from raw input.

Using git:
```
1. Pull from remote: git pull origin master
2. Check status: git status
3. Add files: git add .
4. Commit changes: git commit -m "What changes you made"
5. Push it to remote: git push origin master
6. If some files are deleted, before step 3 do: git add -u .
```
