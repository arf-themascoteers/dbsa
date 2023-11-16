from evaluator import Evaluator

if __name__ == "__main__":
    for dwt in [True,False]:
        for indexify in ["sigmoid","relu"]:
            for retain_relative_position in [True, False]:
                for random_initialize in [True, False]:
                    c = Evaluator()
                    r2, rmse = c.process(dwt,indexify, retain_relative_position,random_initialize)
                    print("r2",round(r2,5))
                    print("rmse",round(rmse,5))