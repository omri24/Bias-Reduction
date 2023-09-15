# Bias-Reduction
# General Information
For this algorithm, bias refers to the diference between the average of predictions of a NN and the real values we try to predict.
This algorithm receives as an input a 2 csv files:

exported_train = a csv that contains the outputs of a NN after passing all the training data into it. first column contains the outputs of the network and the second one contains the real values.

exported_test = a csv that contains the outputs of a NN after passing all the testing data into it. first column contains the outputs of the network and the second one contains the real values.

--- ABOUT THE GIVEN CSV FILES AND THE IMAGES ---
The example csv files here are the data from the EU-USD algorithm, that can be found in the repositories section in my github.
The image files show the estimations before and after applying this algorithm, as well as the precise amount of bias that has been reduced (spoiler: not so much).


--- NOTES ---

Please note that this algorithm will only work for NN that have a single number as an output.
As can be seen in the example files, this bias reduction can be minor.
This algorithm is general and can be helpful for any NN that predicts with a bias. 
