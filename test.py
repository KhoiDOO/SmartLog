# from SmartMachineLearning import SmartML
# print(dir(SmartML))

# from SmartMachineLearning.SmartML import SmartLog
# print(dir(SmartLog))

# from SmartMachineLearning.Metrics.Classification import cm
# print(dir(cm))

# from SmartMachineLearning.Metrics.Classification.cm import CM, deep_analysis, overall_analysis
# print(dir(CM))
# print(dir(deep_analysis))
# print(dir(overall_analysis))

# from SmartMachineLearning.Metrics.Classification.report import Report
# print(dir(Report))

# from SmartMachineLearning.Metrics.Classification.roc_curve import roc_curve_score
# print(dir(roc_curve_score))

# from SmartMachineLearning.Others.utils import to_categorical
# print(dir(to_categorical))

# from SmartMachineLearning.Others.roc_curve_drawing import roc_curve_drawing
# print(dir(roc_curve_drawing))

# from SmartMachineLearning.SmartML import SmartLog
# from random import randint
# import numpy as np
# sl = SmartLog(2, 1, 5, classes=[0, 1])
# y_true = np.array([randint(0, 1) for x in range(100)])
# y_pred = np.array([randint(0, 1) for x in range(100)])
# for x in range(6):
#     for y in range(3):
#         sl.add_results(y_true=y_true, y_pred=y_pred, params_dict={0 : 1, 1 : 0})
# sl.to_json()

# from SmartMachineLearning.Training import Monitor
# print(dir(Monitor))

# from SmartMachineLearning.Training.Monitor import SmartTraining
# print(dir(SmartTraining))

# from SmartMachineLearning.Training.Classification.ensemble import SmartRandomForest
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, 
#                            n_features=10, 
#                            n_classes=2)

# test_params_dict = {
#     "n_estimators" : [50, 100],
#     "criterion" : ["gini"],
#     "max_features" : ["sqrt"],
#     "bootstrap" : [True],
#     "class_weight" : ["balanced", "balanced_subsample"]}

# srf = SmartRandomForest(number_class=2,
#                         number_label=1,
#                         number_fold=5,
#                         classes=[0, 1],
#                         X_data=X,
#                         y_data=y,
#                         params_dict=test_params_dict)
# srf.smartfit()
# srf.to_json()

# from SmartMachineLearning.Training.Classification.linear_poly import SmartLogisticRegression
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, 
#                            n_features=10, 
#                            n_classes=2)

# test_params_dict = {
#     "penalty" : ["l2"],
#     "fit_intercept" : [True, False],
#     "class_weight" : ["balanced", None],
#     "solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}

# srf = SmartLogisticRegression(number_class=2,
#                         number_label=1,
#                         number_fold=5,
#                         classes=[0, 1],
#                         X_data=X,
#                         y_data=y,
#                         params_dict=test_params_dict)
# srf.smartfit()
# srf.to_json()

from SmartMachineLearning.Training.Classification.linear_poly import SmartSupportVectorMachine
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, 
                           n_features=10, 
                           n_classes=2)

srf = SmartSupportVectorMachine(number_class=2,
                        number_label=1,
                        number_fold=5,
                        classes=[0, 1],
                        X_data=X,
                        y_data=y)
srf.smartfit()
srf.to_json()