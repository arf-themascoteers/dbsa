from ds_manager import DSManager
from ann_vanilla import ANNVanilla

if __name__ == "__main__":
    for dwt in [True, False]:
        for indexify in ["sigmoid","relu"]:
            for retain_relative_position in [True, False]:
                for random_initialize in [True, False]:
                    for uniform_lr in [True, False]:
                        ds = DSManager(dwt)
                        train_x, train_y, test_x, test_y, validation_x, validation_y = ds.get_datasets()
                        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
                        ann = ANNVanilla(train_x, train_y, test_x, test_y, validation_x, validation_y,
                                         dwt, indexify, retain_relative_position, random_initialize, uniform_lr)
                        ann.train()