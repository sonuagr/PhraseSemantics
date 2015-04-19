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

How are word-vectors made?:  The meaning of a word is encoded as a vector
computed from co-occurrence statistics of a word and its neighboring words. How?

Paper:
The vector captures the meaning of
that constituent. The matrix captures how it modifies
the meaning of the other word that it combines with.
A representation for a longer phrase is computed
bottom-up by recursively combining the words according 
to the syntactic structure of a parse tree. How did you get the parse tree?
Neural network act as the merging function.
The MV-RNN is the only model that is able to properly negate sen-
timent when adjectives are combined with not.

Next task:
1. See how t for each node is obtained to get cross-entropy error.
2. Change it to the code for movie rating.

TODO for presentation:
1. Write words on paper: L-BFGS: Limited-Memory BFGS which efficiently solves unconstrained nonlinear optimization problems. It can efficiently solve the objective function.
2. cross-entropy function: -l(x)log(d)
3. PCFG parser from stanford.
4. 9 different classes:Cause-Effect,  Instrument-Agency, Product-Producer, Content-Container, Entity-Origin,  Entity-Destination, Component-Whole, Member-Collection, Message-Topic, others
5. pre-trained 50-dimensional word vectors from the unsupervised model of Collobert and Weston

Presentation:
1. How is parse tree made? What is its significance? Why Binary parse tree?
2. Audience shouldn't consider the next parse tree same as the binary parse tree. You can say "Now lets take the statement "a beautiful movie...". Now here..."
3. Look at the pdf attachments for more information.
4. Further improvements: use glove.

Our goal is to find the semantic relation of one noun with other noun of a given sentence. For example in Sentence one, ____________. 

A sentence Parse tree captures the way human speaks. It branches out main phrase into subphrases or specifiers and their complements based on syntactic structure. Now to simplify our task, we would restrict our parse tree to binary.
Now if we could represent each leaf node with a vector which expresses its inherent meaning and also a matrix which shows its power to transform the neighbouring words and if we have a non-linear function which can combine the two child nodes to get the vector representing the phrase containing those two subphrases and matrix for the parent node showing its power to transform the other words in the sentence then we can do it recursively to get the vector representation of the root node which represents the semantics of the whole sentence. 
Now to learn the vector, matrix and also the non linear function, we use Recursive neural networks. To keep the function non-linear, we can use the sigmoid or tanh with a n*2n matrix that combines the two vectors and other n*2n matrix that combines the two matrices.
Example lets consider the sentence: "SpiderMan is a very good movie to watch". Now in this case, the binarized parse tree that we get is something like this. Here, the vector representing the word "very" may be close to zero vector as it doesn't have any significant meaning but its matrix is significant enough to positively modify the word "good".
Now while applying the non-linear function, matrix of b modifies the vector of A and matrix of A modifies the vector of B on which n*2n matrix is applied which is then passed to non-linear function to get the parent vector p. For geting parrent matrix, n*2n matrix Wm is applied on matrices of A and B.
Now, to get the label we use softmax classifier on root node. The objective function is obtained using E which is the sum of cross-entropy errors at all nodes. This is the derivative of our objective function.
Now lets consider the sentence "All the movies showed wars but non was entertaining." To find the relation between "movie" and "wars" we only use the syntactic path between the two and therefor the root of smallest subtree containg them is used to find the relation. For example,_______________, and the vector representation of this node is used to get the classification.
This is the F1-Score:82.5% and accuracy of 77%. The dataset we used is SemEval 2010 task 8.
Our next target would be to pretrain the words using GloVe.

What can be done:
1. Tag first and last word and find the relation between them which will use syntactic path between them. Use this to find the semantics of the sentence. Will it work?
2. Use above one to find he semantics of the whole movie review: 
vector: [lengthRatioOfSentence1wrtParaLength*outputForSentence1+lengthRatioOfSentence2wrtParaLength*outputForSentence2+.....] used to learn with logisticRegression.

Now using a nonlinear function

Using git:
```
1. Pull from remote: git pull origin master
2. Check status: git status
3. Add files: git add .
4. Commit changes: git commit -m "What changes you made"
5. Push it to remote: git push origin master
6. If some files are deleted, before step 3 do: git add -u .
```

Feadback:
Try to see in which ways you can explore alternative ideas.
Please try to use on some other data.

Positive sentences: 5331
Negative sentences: 5331
data/RTData_CV1.mat: data of negative + positive sentences
output/resultsRAE.txt: output of running test/train
output/optParams_RT_CV1.mat: Some parameters used for testing like opttheta. 
Train: 10662-1066 (Trained only on 9/10 of datas)
Test: 10662

What are the no of epochs?
(.*)[a-zA-Z]*[-][a-zA-Z]*[(](e1,e2)[)](.*)
allSStr contains all the training strings in allDataTrain.mat

Training Socher RNN:
initParams.m:
	params.paths.data = '../dataCamera/';
	params.paths.outputFolder = '../output/emnlp/';
	params.paths.results = [params.paths.outputFolder 'results/'];
loadData.m loads dataCamera/allDataTrain.mat without loading parameter: "external"

After optimizing, the result is tested with(out) externel features and stored at 
[params.paths.outputFolder fileName '.mat'] where fileName: 'weights_WEF_acc_' num2str(F1) or 'weights_WOEF_acc_' num2str(F1) which can directly be used for testing by classifyrelations.sh.
The file it uses is allDataTest.mat which is the result of testing the test data :P