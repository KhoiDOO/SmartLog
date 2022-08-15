from sklearn.metrics import confusion_matrix

class CM:
    def __init__(self, y_true, y_pred):
        self.cm = confusion_matrix(y_true, y_pred) # confusion matrix
        self.TP = self.cm[0][0] # true positive
        self.FN = self.cm[1][0] # false negative
        self.FP = self.cm[1][0] # false positive
        self.TN = self.cm[1][1] # true negative
    
    def get_true_positive(self):
        return self.TP

    def get_false_positive(self):
        return self.FP

    def get_true_negative(self):
        return self.TN

    def get_false_negative(self):
        return self.FN

    def get_cm(self, dict_form):
        if(dict_form):
            return {
                "TP" : self.get_true_positive(),
                "FP" : self.get_false_negative(),
                "TN" : self.get_true_negative(),
                "FN" : self.get_false_negative()
            }
        else:
            return self.cm