# Fingerprint Spoofing Detection

For detailed description of the experiments performed during this project, refer to `MLPR_fingerprint_spoofing_detection.pdf`.

## Models implemented

- MVG
- Logistic regression
- SVM
- GMM

## Run training

Folders `mvg`, `logreg`, `svm` and `gmm` each contain a `train_*.py` script (e.g. `gmm/train_gmm.py`).

For `mvg/train_mvg.py` and `gmm/train_gmm.py`, it's enough to run the scripts as they are.

For `svm/train_svm.py`, there are the 3 functions `train_svm()`, `train_svm_poly()` and `train_svm_rbf()`. To run one of them, uncomment the wanted one inside the `if __name__ == '__main__'` block.

For `logreg/train_logreg.py`, at line 36, there is `expand_feat = True`. To train linear logistic regression, set `expand_feat = False`. For quadratic logistic regression, set `expand_feat = True`.

The training functions save the parameters and the scores obtained on the training set under all the hyperparameters configurations tried. These are stored in subfolders `results` or `results_*` (e.g. `svm/results_poly`).

## Run evaluation

`./evaluation.py` contains the functions to test the trained model on the evaluation set. In `if __name__ == '__main__'` block, uncomment the desired function to be called.

Evaluation results saved in `evaluation_logs`.

## Run calibration

`./evaluation.py` contains function for calibrating the scores of the best models chosen during training and validation (functions `calibration_on_validation_data()` and `calibration_on_eval_data()`). Within each function, uncomment the lines corresponding to the model for which you want to plot the Bayes Error Plot.

## Plot performance

All plot functions are in `utils/plot_utils.py`
