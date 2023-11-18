import pandas as pd
import pywt
import matplotlib.pyplot as plt

df = pd.read_csv("data/dataset_min.csv")
start_index = list(df.columns).index("400")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
axes = axes.flatten()

signal = df.iloc[0].to_numpy()
signal = signal[start_index:]
axes[0].plot(signal)
axes[0].set_title("Sample 1")
annotated_point = (600, signal[600])
axes[0].annotate(f'Band#{600}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))
annotated_point = (3000, signal[3000])
axes[0].annotate(f'Band#{3000}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))

signal = df.iloc[10].to_numpy()
signal = signal[start_index:]
axes[1].plot(signal)
axes[1].set_title("Sample 2")
annotated_point = (600, signal[600])
axes[1].annotate(f'Band#{600}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))
annotated_point = (3000, signal[3000])
axes[1].annotate(f'Band#{3000}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))


signal = df.iloc[101].to_numpy()
signal = signal[start_index:]
axes[2].plot(signal)
axes[2].set_title("Sample 3")
annotated_point = (600, signal[600])
axes[2].annotate(f'Band#{600}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))
annotated_point = (3000, signal[3000])
axes[2].annotate(f'Band#{3000}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))


signal = df.iloc[1013].to_numpy()
signal = signal[start_index:]
axes[3].plot(signal)
axes[3].set_title("Sample 4")
annotated_point = (600, signal[600])
axes[3].annotate(f'Band#{600}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))
annotated_point = (3000, signal[3000])
axes[3].annotate(f'Band#{3000}', xy=annotated_point, xytext=annotated_point,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()


# signal,_,_,_,_,_,_ = pywt.wavedec(signal, 'db1', level=6)
# print(signal.shape)
# plt.plot(signal)
# plt.show()

