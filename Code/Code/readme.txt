Semantic Compositionality through Recursive Matrix-Vector Spaces
================================================================
+------------------------------------------------------------------------------------------+
| 1. Setting up:                                                                           |
+------------------------------------------------------------------------------------------+

We use slightly modified code of Socher. 
1. Download relationClassification.zip from:
	http://nlp.stanford.edu/~socherr/relationClassification.zip
2. Unzip relationClassification.zip
3. Make sure that following are installed before using the code:
	1. Python with numpy and scipy
	2. Matlab
4. Copy deepDualCamera/trainDual.m to relationClassification/deepDualCamera/

+------------------------------------------------------------------------------------------+
| 2. Testing and Training on SemEval2010 Task 8:                                           |             
+------------------------------------------------------------------------------------------+

Refer relationClassification/README.txt

+------------------------------------------------------------------------------------------+
| 2. Testing and Training on SemEval2007 Task 4:                                           |
+------------------------------------------------------------------------------------------+

For Testing:
1. Copy classifyRelations/weights_WOEF.mat to relationClassification/classifyRelations/
2. Go to relationClassification/classifyRelations/
3. Run "classifyRelations.sh  <input text file>  <output mat file>""

For Training and Testing:
1. Copy dataCamera/allDataTrain.mat to relationClassification/dataCamera/
2. Copy dataCamera/allDataTest.mat to relationClassification/dataCamera/
3. Go to relationClassification/dataCamera/
4. Run "trainDual" in matlab

+------------------------------------------------------------------------------------------+
| 3. SemEval2007 Task 4 Training and Test files:                                           |
+------------------------------------------------------------------------------------------+

The labeled train and test files are located at SemEval2007Task4_modified

+------------------------------------------------------------------------------------------+
| 4. Hacks:                                                                                |
+------------------------------------------------------------------------------------------+

1. How training mat files are obtained:

allDataTrain.mat is obtained by testing FILE_FULL_TRAIN2007.txt on present socher's code.
allDataTest.mat is obtained by testing FILE_FULL_TEST2007.txt on present socher's code.

2. How testing mat file is obtained:

After training on trainDual, output/emnlp/weights_WOEF_acc_F1.mat is generated which can be 
used as classifyRelations/weights_WOEF.mat
