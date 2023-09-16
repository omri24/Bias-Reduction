# Bias-Reduction
## General information
For this algorithm, bias refers to the diference between the average of predictions of a NN and the real values tried to predict.
This algorithm receives as an input a 2 csv files:
1. exported_train = a csv that contains the outputs of a NN after passing all the training data into it. first column contains the outputs of the network and the second one contains the real values.
2. exported_test = a csv that contains the outputs of a NN after passing all the testing data into it. first column contains the outputs of the network and the second one contains the real values.
The example CSV files here are the data from the EU-USD algorithm, that can be found in the repositories section in my Github.

## The results
The images show the estimations before and after applying this algorithm, as well as the precise amount of bias that has been reduced (spoiler: not so much).

Estimations before bias reduction:

![estimations before bias reduction algorithm](https://github.com/omri24/Bias-Reduction/assets/115406253/b56e9524-12aa-4e18-bf10-e069276761aa)

Estimations after bias reduction:

![estimations after applying the bias reduction algorithm](https://github.com/omri24/Bias-Reduction/assets/115406253/e5cd714b-36d2-4c94-a7f9-40916763da8f)

Bias improvement:

![output numbers of the algorithm](https://github.com/omri24/Bias-Reduction/assets/115406253/51c423b2-6fd0-4a0f-a3d6-e495c3709737)

## Notes

Please note that this algorithm will only work for NN that have a single number as an output.
The bias reduction achieved by this algorithm can be minor.
This algorithm is general and can be helpful for any NN that predicts with a bias. 
