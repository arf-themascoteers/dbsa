import matplotlib
matplotlib.use("TkAgg")
import math
import matplotlib.pyplot as plt
import pandas as pd


def plot_me_plz(dwt=True,indexify="sigmoid", retain_relative_position=True,random_initialize=True):
    file = f"{str(dwt)}_{indexify}_{str(retain_relative_position)}_{str(random_initialize)}.csv"
    df = pd.read_csv(file)
    epoch_col = df.columns.get_loc("epoch")
    rw_col = df.columns.get_loc("r2")
    rmse_col = df.columns.get_loc("rmse")
    itrs = len(df)
    sis = [{"name":"r2"},{"name":"rmse"}]
    si = None
    for index in range(rmse_col+1,len(df.columns)):
        col = df.columns[index]
        if "#" in col:
            if si is None:
                si = {}
            else:
                sis.append(si)
                si = {}
            si["name"] = col
            si["display_name"] = df.iloc[0,index]
            si["params"] = []
        else:
            si["params"].append(col)
    sis.append(si)
    print(sis)
    total_plots = len(sis)
    rows = math.ceil(total_plots/4)

    fig, axes = plt.subplots(nrows=1, ncols=3)

    axes = axes.flatten()
    band_serial = 1
    for i,p in enumerate(sis):
        name = p["name"]
        if name in ["r2"]:
            ax = axes[0]
            data = df[name].tolist()
            ax.plot(data)
            ax.set_title(name)
        elif name in ["rmse"]:
            ax = axes[1]
            data = df[name].tolist()
            ax.plot(data)
            ax.set_title(name)
        else:
            ax = axes[2]
            for index,a_param in enumerate(p["params"]):
                data= df[a_param].tolist()
                ax.plot(data, label = f"Band-{band_serial}")
                ax.set_ylim(1, 4300)
                band_serial = band_serial+1
    axes[2].legend(loc='center right', framealpha=0.1)
    axes[2].set_title("Bands")

    plt.show()


if __name__ == "__main__":
    plot_me_plz()
