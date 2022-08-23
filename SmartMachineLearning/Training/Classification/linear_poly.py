from SmartMachineLearning.Training.Monitor import SmartTraining
from sklearn.linear_model import LogisticRegression
from datetime import datetime

class SmartLogisticRegression(SmartTraining):
    def __init__(self, 
                 number_class=2, 
                 number_label=1, 
                 number_fold=5, 
                 problem_type="Classification", 
                 classes=None, 
                 X_data=None, 
                 y_data=None, 
                 test_size=0.33, 
                 random_state=42,
                 params_dict = {
                     "penalty" : ["l2"],
                     "fit_intercept" : [True, False],
                     "class_weight" : ["balanced", None],
                     "solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                     }):
        super().__init__(number_class, 
                         number_label, 
                         number_fold, 
                         problem_type, 
                         classes, 
                         X_data, 
                         y_data, 
                         test_size, 
                         random_state)
        self.params_dict = params_dict
    
    def training(self, penalty, 
                 fit_intercept, 
                 class_weight, 
                 solver,
                 njob):
        fold_index = 0
        for train_index, test_index in self.kf.split(self.X_data):
            print("\tFold: {}".format(fold_index))
            print("\tTRAIN:", f"{train_index[0]} -- {train_index[-1]}", 
                  "\n\tTEST:", f"{test_index[0]} -- {test_index[-1]}")
        
            # folding data
            X_train, X_test = self.X_data[train_index], self.X_data[test_index]
            y_train, y_test = self.y_data[train_index], self.y_data[test_index]
    
            # Training
            print("\t\tTraining : {}".format(fold_index), end = " -- ")
            print("Start: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")), end=" --- ")
            model = LogisticRegression(penalty = penalty,
                                         fit_intercept = fit_intercept,
                                         class_weight = class_weight,
                                         solver = solver,
                                          n_jobs = njob)
            model.fit(X_train,y_train)
            print("End: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        
            # Testing
            print("\t\tValidation: {}".format(fold_index), end = " -- ")
            print("Start: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")), end=" --- ")
            y_pred = model.predict(X_test)
            print("End: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        
            # Evaluation
            base_params = {"penalty" : penalty, 
                           "fit_intercept" : fit_intercept, 
                           "class_weight" : class_weight, 
                           "solver" : solver}
            self.add_results(y_true = y_test, y_pred = y_pred, params_dict = base_params)
            fold_index += 1
        
    def smartfit(self):
        count = 0
        for x in self.params_dict["penalty"]:
            for i in self.params_dict["fit_intercept"]:
                for j in self.params_dict["class_weight"]:
                    for k in self.params_dict["solver"]:
                            print("Traning Case: {}".format(count))
                            self.training(x, i, j, k, 5)
