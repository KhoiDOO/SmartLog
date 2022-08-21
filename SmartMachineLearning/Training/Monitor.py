# from SmartML import SmartLog
from SmartMachineLearning.SmartML import SmartLog
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class SmartTraining(SmartLog):
    def __init__(self, number_class = 2, number_label = 1, number_fold = 5, problem_type="Classification", classes=None,
                 X_data = None, y_data = None, test_size = 0.33, random_state = 42):
        super().__init__(number_class, number_label, number_fold, problem_type, classes)
        if(not X_data):
            print("X data can not be None")
        elif(not y_data):
            print("y_data can not be None")
        else:
            self.X_data = X_data
            self.y_data = y_data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, 
                                                                                    y_data, 
                                                                                    test_size=test_size, 
                                                                                    random_state=random_state) 
        self.kf = KFold(n_split = number_fold)
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_train(self):
        return self.y_train
    
    def get_y_test(self):
        return self.y_test
    
    def get_k_fold_info(self):
        return self.kf