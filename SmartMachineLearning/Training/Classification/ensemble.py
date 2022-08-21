from SmartMachineLearning.Training.Monitor import SmartTraining
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class SmartRandomForest(SmartTraining):
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
                     "n_estimators" : [50, 100, 150, 200],
                     "criterion" : ["gini", "entropy", "log_loss"],
                     "max_features" : ["sqrt", "log2", None],
                     "bootstrap" : [True, False],
                     "class_weight" : ["balanced", "balanced_subsample", None]
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
    
    def training(self, n_estimators, criterion, max_features, bootstrap, class_weight, njob):
        fold_index = 0
        for train_index, test_index in self.kf.split(self.X):
            print("\tFold: {}".format(fold_index))
            print("\tTRAIN:", train_index, "\n\tTEST:", test_index)
        
            # folding data
            X_train, X_test = self.X_data[train_index], self.X_data[test_index]
            y_train, y_test = self.y_data[train_index], self.y_data[test_index]
    
            # Training
            print("\t\tTraining : {}".format(fold_index), end = " -- ")
            print("Start: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")), end=" --- ")
            model_RF = RandomForestClassifier(n_estimators = n_estimators, 
                                              criterion = criterion, 
                                              max_features = max_features, 
                                              bootstrap = bootstrap,
                                              class_weight = class_weight, 
                                              n_jobs = njob)
            model_RF.fit(X_train,y_train)
            print("End: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        
            # Testing
            print("\t\tValidation: {}".format(fold_index), end = " -- ")
            print("Start: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")), end=" --- ")
            y_pred = model_RF.predict(X_test)
            print("End: {}".format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        
            # Evaluation
            base_params = {"n_estimators" : n_estimators, 
                           "criterion" : criterion, 
                           "max_features" : max_features, 
                           "bootstrap" : bootstrap, 
                           "class_weight" : class_weight}
            self.add_results(y_true = y_test, y_pred = y_pred, params_dict = base_params)
            fold_index += 1
        
    def smartfit(self):
        count = 0
        for x in self.param_grid["n_estimators"]:
            for i in self.param_grid["criterion"]:
                for j in self.param_grid["max_features"]:
                    for k in self.param_grid["bootstrap"]:
                        for l in self.param_grid["class_weight"]:
                            print("Traning Case: {}".format(count))
                            self.training(x, i, j, k, l, 5)  
    
    
    
    
