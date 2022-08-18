from unicodedata import decimal
import numpy as np
import random

# Test Sample
y_true = np.array([random.randint(0, 1) for x in range(100)])
y_pred = np.array([random.randint(0, 1) for x in range(100)])

# Confusion Matrix Test
# from Classification.cm import CM
# cm_test = CM(y_true=y_true, y_pred=y_pred)
# print(cm_test.get_cm(dict_form=True))

# Confusion Matrix - Overall Analysis Test
# from Classification.cm_overall_analysis import overall_analysis
# cm_test = overall_analysis(y_true=y_true, y_pred=y_pred)
# print(cm_test.get_overall_analysis())

# Confusion Matrix - Deep Analysis
# from Classification.cm_deep_analysis import deep_analysis
# cm_test = deep_analysis(y_true=y_true, y_pred=y_pred)
# print(cm_test.get_deep_analysis())

#Roc curve test
# from Classification.roc_curve import roc_curve_score
# roc_test = roc_curve_score(y_true, y_pred, 2)
# print(roc_test.get_roc_dict())

# Report test
# from Classification.report import Report
# report_test = Report(y_true=y_true, y_pred=y_pred)
# print(report_test.get_cls_report())
# print(report_test.get_roc_auc_dict(full=False, type='macro'))
# print(report_test.get_roc_auc_dict())

# Roc curve drawing test
# from Others.roc_curve_drawing import roc_curve_drawing
# from Classification.roc_curve import roc_curve_score
# roc_curve_score_test = roc_curve_score(y_true=y_true, y_pred=y_pred, num_class=2)
# fprs = roc_curve_score_test.get_fpr()
# tprs = roc_curve_score_test.get_tpr()
# roc_curve_drawing_test = roc_curve_drawing()
# roc_curve_drawing_test.draw_curve(fprs, tprs, title = "ROC CURVE TEST", num_class=2)

# History Test
# import SmartLog as sl
# from SmartLog import SmartLog
# SmartLogTest = SmartLog(2, 1, 5, classes=[0, 1])
# for i in range(6):
#     SmartLogTest.add_results(y_true=y_true, y_pred=y_pred, params_dict={0 : 1, 1 : 0})
# print(SmartLogTest.get_history())
# SmartLogTest.to_json()



# Test Smart Training
from SmartLog import SmartTraining
_dict = {
    "n_estimators" : [50, 100, 150, 200],
    "criterion" : ["gini", "entropy", "log_loss"],
    "max_features" : ["sqrt", "log2", None],
    "bootstrap" : [True, False],
    "class_weight" : ["balanced", "balanced_subsample", None]
}

st = SmartTraining(2, 1, 5, classes=[0, 1], grid_params=_dict)
