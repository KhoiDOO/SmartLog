from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

class Report:
    def __init__(self, y_true, y_pred):
        self.roc_auc_macro = roc_auc_score(y_true, y_pred)
        self.roc_auc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
        self.roc_auc_weighted = roc_auc_score(y_true, y_pred, average = 'weighted')
        self.cls_report = classification_report(y_true, y_pred, output_dict=True)
    
    def get_roc_auc(self, type):
        switcher = {
            "macro": self.roc_auc_macro,
            "micro": self.roc_auc_micro,
            "weighted": self.roc_auc_weighted,
        }

        return switcher.get(type, "incorrect type")
    
    def get_cls_report(self):
        return self.cls_report
    
    def get_roc_auc_dict(self, full = True, type = None):
        if(full):
            return {
                "macro": self.roc_auc_macro,
                "micro": self.roc_auc_micro,
                "weighted": self.roc_auc_weighted,
            }
        else:
            if(type):
                return self.get_roc_auc(type)
            else:
                return "Type needed including"
