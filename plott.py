import matplotlib.pyplot as plt
import pandas as pd


def plot_me_plz(filename):
    df = pd.read_csv(filename)
    all_columns = list(df.columns)
    band_columns = [col for col in all_columns if "band" in col]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes = axes.flatten()

    axes[0].plot(df["train_r2"].tolist(), label="train_r2")
    axes[0].plot(df["test_r2"].tolist(), label="test_r2")
    axes[0].plot(df["validation_r2"].tolist(), label="validation_r2")
    axes[0].set_title("R2")

    axes[1].plot(df["train_rmse"].tolist(), label="train_rmse")
    axes[1].plot(df["test_rmse"].tolist(), label="test_rmse")
    axes[1].plot(df["validation_rmse"].tolist(), label="validation_rmse")
    axes[1].set_title("RMSE")

    axes[2].plot(df["time"].tolist(), label="time")
    axes[2].set_title("Time")

    for band in band_columns:
        axes[3].plot(df[band].tolist(), label=band)
    axes[3].set_title("Bands")
    axes[3].legend(loc='center right', framealpha=0.1)

    plt.show()


if __name__ == "__main__":
    plot_me_plz("True_sigmoid_True_True.csv")
