from sklearn.metrics import confusion_matrix
import math

class CM:
    def __init__(self, y_true, y_pred):
        self.cm = confusion_matrix(y_true, y_pred) # confusion matrix
        self.TP = int(self.cm[1][1]) # true positive
        self.FN = int(self.cm[1][0]) # false negative
        self.FP = int(self.cm[0][1]) # false positive
        self.TN = int(self.cm[0][0]) # true negative
    
    def get_true_positive(self):
        return self.TP

    def get_false_positive(self):
        return self.FP

    def get_true_negative(self):
        return self.TN

    def get_false_negative(self):
        return self.FN

    def get_cm(self, dict_form = True):
        if(dict_form):
            return {
                "TP" : self.get_true_positive(),
                "FP" : self.get_false_negative(),
                "TN" : self.get_true_negative(),
                "FN" : self.get_false_negative()
            }
        else:
            return self.cm

class deep_analysis(CM):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
    
    def sensitivity(self):
        return self.TP / (self.TP + self.FN)

    def specificity(self):
        return self.TN / (self.TN + self.FP)    

    def precision(self):
        return self.TP/(self.FN +  self.TP)

    def negative_predictive_value(self):
        return self.TN / (self.TN + self.FN) 
    
    def false_negative_rate(self):
        return self.FN / (self.FN + self.TP)

    def false_positive_rate(self):
        return self.FP / (self.FP + self.TN)

    def false_discovery_rate(self):
        return self.FP / (self.FP + self.TP)

    def false_omission_rate(self):
        return self.FN / (self.FN + self.TN)

    def positive_likelihood_ratio(self):
        return self.sensitivity() / self.false_positive_rate()
    
    def negative_likelihood_ratio(self):
        return self.false_negative_rate() / self.specificity()

    def prevalence_threshold(self):
        return math.sqrt(self.false_positive_rate()) / (math.sqrt(self.sensitivity()) + math.sqrt(self.false_positive_rate()))

    def threat_score(self):
        return self.TP / (self.TN + self.FN + self.FP)

    def prevalence(self):
        return (self.TP + self.FN)/(self.TP + self.FN + self.TN + self.FP)
    
    def matthews_correlation_coefficient(self):
        return (self.TP*self.TN - self.FN*self.FP) / ((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))

    def fowlkes_mallows_index(self):
        return math.sqrt(self.sensitivity() + self.precision())

    def informedness(self):
        return self.sensitivity() + self.specificity() - 1
    
    def markedness(self):
        return self.precision() + self.negative_predictive_value()
    
    def diagnostic_odds_ratio(self):
        return self.positive_likelihood_ratio() / self.negative_likelihood_ratio()

    def get_deep_analysis(self):
        return {
            "negative_predictive_value" : self.negative_predictive_value(),
            "false_negative_rate" : self.false_negative_rate(),
            "false_positive_rate" : self.false_positive_rate(),
            "false_discovery_rate" : self.false_discovery_rate(),
            "false_omission_rate" : self.false_omission_rate(),
            "positive_likelihood_ratio" : self.positive_likelihood_ratio(),
            "negative_likelihood_ratio" : self.negative_likelihood_ratio(),
            "prevalence_threshold" : self.prevalence_threshold(),
            "threat_score" : self.threat_score(),
            "prevalence" : self.prevalence(),
            "matthews_correlation_coefficient" : self.matthews_correlation_coefficient(),
            "fowlkes_mallows_index" : self.fowlkes_mallows_index(),
            "informedness" : self.informedness(),
            "markedness" : self.markedness(),
            "diagnostic_odds_ratio" : self.diagnostic_odds_ratio()
        }


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