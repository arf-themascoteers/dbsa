from ds_manager import DSManager
import os


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results", "results.csv")
        self.columns = [
            "features",
            "samples",

            "r2",
            "rmse",
            "selected_params"
            
            "sis",
        ]
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as file:
                file.write(",".join(self.columns) + "\n")

    def evaluate(self):
        for task in self.tasks:
            print("*****************************************")
            print(task)
            print("*****************************************")
            feature = task["feature"]
            sample = task["sample"]
            sis = task["sis"]
            dataset = DSManager(feature, sample)
            r2, rmse, selected_params = self.process(dataset, sis)
            sis_str = str(sis)
            sis_str = sis_str.replace(",",";")
            selected_params_str = str(selected_params)
            selected_params_str = selected_params_str.replace(",",";")
            with open(self.filename, 'a') as file:
                file.write(
                    f"{dataset.count_features()},"
                    f"{sample},"
                    
                    f"{r2},"                    
                    f"{rmse},"                    
                    f"{selected_params_str},"
                    
                    f"{sis_str}")

    def process(self, dataset, sis):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        r2, rmse, selected_params = algorithm.predict_it(X_test, y_test)
        return r2, rmse, selected_params