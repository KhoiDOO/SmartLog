class SmartLog:
    def __init__(self, number_class, number_label, fold_count, problem_type):
        """
            - number_class: number of class in the data set
            - number_label: numner of label in the data set
            - fold_count: number of fold in each train case
            - problem_type: type of problem facing
        """
        
        pass

    def add(self, y_true, y_pred, params_dict, train_index, fold_index):
        """
            - y_true: ground truth of test data
            - y_pred: output of the model
            - params_dict: dictionary of hyperparameter of the model
            - train_index: the curent index of training
            - fold_index: the current fold of training
        """
        
        pass

    def get_history(self):
        pass

    def save_history(self):
        pass