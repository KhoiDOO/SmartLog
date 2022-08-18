from Classification.cm import CM
from Classification.cm_deep_analysis import deep_analysis
from Classification.cm_overall_analysis import overall_analysis
from Classification.report import Report
from Classification.roc_curve import roc_curve_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import numpy as np

class SmartLog:
    def __init__(self, number_class, number_label, number_fold, problem_type = "Classification", classes = None):
        self.num_class = number_class
        self.num_label = number_label
        self.num_fold = number_fold
        self.problem_type = problem_type
        self.classes = classes
        self.current_train_index = 0
        self.current_fold_index = 0
        self.train_log = {
            'confusion_matrix' : {},
            'overall_analysis' : {},
            'deep_analysis' : {},
            'ROC_CURVE' : {},
            'AUC_SCORE' : {},
            'Classification_Report' : {}
            }
        
        self.history = {
            "overall_info" : {
                "num_class" : self.num_class,
                "num_label" : self.num_label,
                "num_fold" : self.num_fold,
                "problem_type" : self.problem_type,
                "classes" : self.classes
            },
            "history_data" : {
                0 : {
                    "params_dict" : {},
                    "fold_data" : {
                        x : self.train_log for x in range(self.num_fold)
                        }
                    }
                }                
            }
        
    def add_results(self, y_true, y_pred, params_dict):
        if(self.current_fold_index == self.num_fold):
            self.current_train_index += 1
            self.history["history_data"][self.current_train_index] = {
            "params_dict" : params_dict,
            "fold_data" : {
                x : self.train_log for x in range(self.num_fold)
                }
        }
        else:
            cm = CM(y_true=y_true, y_pred=y_pred).get_cm(dict_form=True)
            deep_ana = deep_analysis(y_true=y_true, y_pred=y_pred).get_deep_analysis()
            overall_ana = overall_analysis(y_true=y_true, y_pred=y_pred).get_overall_analysis()
            roc_data = roc_curve_score(y_true=y_true, y_pred=y_pred, num_class=self.num_class).get_roc_dict()
            report = Report(y_true=y_true, y_pred=y_pred)
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['confusion_matrix'] = cm
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['overall_analysis'] = overall_ana
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['deep_analysis'] = deep_ana
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['ROC_CURVE'] = roc_data
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['AUC_SCORE'] = report.get_roc_auc_dict()
            self.history["history_data"][self.current_train_index]["fold_data"][self.current_fold_index]['Classification_Report'] = report.get_cls_report()
            self.current_fold_index += 1
            
    def get_history(self):
        return self.history

    def to_json(self, path = 'json-history.json'):
        import json
        import os
        json_obj = json.loads(json.dumps(self.history, default=str))
        
        with open(path,
                  'w',
                  encoding='utf-8') as outfile:
            json.dump(json_obj, outfile, ensure_ascii=False, indent=4)
        
    def to_pickle(self, path = 'pickle-history.pickle'):
        import pickle
        dbfile = open(path, 'ab')
        pickle.dump(self.history, dbfile)                     
        dbfile.close()
    
    def to_bson(self, path = 'bson-history.bson'):
        pass



            
            
            