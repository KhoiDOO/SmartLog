from Classification.cm import CM

class overall_analysis(CM):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
    
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    
    def balanced_accuracy(self):
        return (self.sensitivity() + self.specificity()) / 2

    def sensitivity(self):
        return self.TP / (self.TP + self.FN)

    def specificity(self):
        return self.TN / (self.TN + self.FP)    

    def precision(self):
        return self.TP/(self.FN +  self.TP)

    def f1_score(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)

    def get_overall_analysis(self):
        return {
            "accuracy" : self.accuracy(),
            "balanced_accuracy" : self.balanced_accuracy(),
            "sensitivity" : self.sensitivity(),
            "specificity" : self.specificity(),
            "precision" : self.precision(),
            "f1_score" : self.f1_score()
        }