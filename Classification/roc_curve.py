from sklearn.metrics import roc_curve
from Others.utils import to_categorical
import numpy as np

class roc_curve_score:
    def __init__(self, y_true, y_pred, num_class):
        self.y_test = to_categorical(y_true)
        self.y_score = to_categorical(y_pred)
        self.fprs = {}
        self.tprs = {}
        self.thresh_holds = {}
        for x in range(num_class):
            self.fprs[x], self.tprs[x], self.thresh_holds[x] = roc_curve(self.y_test[:, x], self.y_score[:, x], drop_intermediate=False)
        
        self.fpr_micro_avg, self.tpr_micro_avg, self.threshold_micro_avg, = roc_curve(self.y_test.ravel(), self.y_score.ravel())
        
        all_fpr = np.unique(np.concatenate([self.fprs[i] for i in range(num_class)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += np.interp(all_fpr, self.fprs[i], self.tprs[i])
        mean_tpr /= num_class
        self.fpr_macro_avg = all_fpr
        self.tpr_macro_avg = mean_tpr
    
    def get_tpr(self, _class = None):
        if(_class):
            return self.tprs[_class]
        else:
            return self.tprs
    
    def get_fpr(self, _class = None):
        if(_class):
            return self.fprs[_class]
        else:
            return self.fprs

    def get_thresholds(self, _class = None):
        if(_class):
            return self.thresh_holds[_class]
        else:
            return self.thresh_holds

    def get_roc_dict(self):
        return {
            "tpr" : self.get_tpr(),
            "fpr" : self.get_fpr(),
            "thresholds" : self.get_thresholds(),
            "fpr_micro_avg" : self.fpr_micro_avg,
            "tpr_micro_avg" : self.tpr_micro_avg,
            "fpr_macro_avg" : self.fpr_macro_avg,
            "tpr_macro_avg" : self.tpr_macro_avg,
        }