from operator import pos
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

