from ds_manager import DSManager
import os
from my_machine import MyMachine


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results", "results.csv")
        self.columns = [
            "features",
            "samples",

            "r2_train",
            "r2_validation",
            "r2_test",
            "rmse_train",
            "rmse_validation",
            "rmse_test",
            "csv_file"
            "sis",
        ]
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as file:
                file.write(",".join(self.columns) + "\n")

    def evaluate(self):
        model_files = []
        for task in self.tasks:
            print("*****************************************")
            print(self.format_tasks(task))
            print("*****************************************")
            feature = task["feature"]
            sample = task["sample"]
            sis = task["sis"]
            lock = False
            if "lock" in task:
                lock = task["lock"]
            dataset = DSManager(feature, sample)
            r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test, machine = \
                self.process(dataset, sis, lock)
            r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test, sis = \
                self.str_process(r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test, sis)
            model_files.append(machine.model_file)
            with open(self.filename, 'a') as file:
                file.write(
                    f"{dataset.count_features()},"
                    f"{sample},"
                    
                    f"{r2_train},"  
                    f"{r2_validation},"
                    f"{r2_test},"                    
                    f"{rmse_train},"                    
                    f"{rmse_validation},"                    
                    f"{rmse_test},"                    
                    f"{machine.csv_file},"
                    f"{sis}\n")
        return model_files

    def process(self, dataset, sis, lock):
        machine = MyMachine(sis, lock)
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test = machine.score(X_train, y_train, X_test, y_test)
        return r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test, machine

    def str_process(self, r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test, sis):
        sis_str = self.format_sis(sis)
        return \
                self.nf(r2_train), \
                self.nf(r2_validation), \
                self.nf(r2_test), \
                self.nf(rmse_train),\
                self.nf(rmse_validation),\
                self.nf(rmse_test),\
                sis_str

    def nf(self, metric):
        return f"{round(metric, 2):.2f}"

    def format_sis(self, sis):
        the_strs = []
        for si in sis:
            si_name = si["si"].__name__
            si_count = str(si["count"])
            the_strs.append(f"{si_name}:{si_count}")
        return ";".join(the_strs)

    def format_tasks(self, tasks):
        formatted = f"Features:{tasks['feature']}, Samples: {tasks['sample']}"
        sis = tasks['sis']
        sis_str = self.format_sis(sis)
        return f"{formatted}, Spectral Indices: {sis_str}"
