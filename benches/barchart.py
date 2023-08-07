# [Mean, Median, StdDev, IQR, Rounds]
# 2d Booth function
test_crust = [72.4802, 72.0070, 2.1507, 0.8380, 30779]
test_scipy = [557.1416, 550.8085, 17.7928, 19.6260, 5148]
test_numpy = [4844.8629, 4848.7410, 34.9357, 32.5105, 440]

# 1d Rastrigen
rast_1d_crust = [111.0920, 110.2810, 6.0843, 1.4670, 24434]
rast_1d_scipy = [596.3572, 597.9870, 23.4832, 26.0338, 4879]
rast_1d_numpy = [4844.6139, 4822.0270, 47.0569, 77.9960, 967]

# 10d Rastrigen
rast_10d_crust = [128.7791, 127.2520, 4.2710, 3.1430, 21100]
rast_10d_scipy = [727.2459, 722.7940, 28.6653, 7.9620, 4356]
rast_10d_numpy = [4820.2199, 4820.4200, 38.9686, 31.0272, 853]

# 100d Rastrigen
rast_100d_crust = [377.0360, 373.1660, 12.7395, 7.0540, 10386]
rast_100d_scipy = [3256.5209, 3288.7860, 223.5415, 285.8620, 1462]
rast_100d_numpy = [4957.5911, 4939.9895, 77.5008, 46.8285, 920]


# Rastrigen Mean [crust, scipy, numpy]
import numpy as np
import matplotlib.pyplot as plt

# Define the data
rast_1d_crust, rast_1d_scipy, rast_1d_numpy = 111.0920, 596.3572, 4844.6139
rast_10d_crust, rast_10d_scipy, rast_10d_numpy = 128.7791, 727.2459, 4820.2199
rast_100d_crust, rast_100d_scipy, rast_100d_numpy = 377.0360, 3256.5209, 4957.5911

# Set the positions and width of the bars
pos_1d = np.arange(3)
pos_10d = [x + 0.25 for x in pos_1d]
pos_100d = [x + 0.5 for x in pos_1d]
bar_width = 0.25

# Create the horizontal bar chart
fig, ax = plt.subplots()
opacity = 0.8

# 1d Rastrigen
ax.barh(
    pos_1d,
    [rast_100d_crust, rast_10d_crust, rast_1d_crust],
    bar_width,
    alpha=opacity,
    color="#1f77b4",
    label="flash_rbf",
)

# 10d Rastrigen
ax.barh(
    pos_10d,
    [rast_100d_scipy, rast_10d_scipy, rast_1d_scipy],
    bar_width,
    alpha=opacity,
    color="#ff7f0e",
    label="scipy",
)

# 100d Rastrigen
ax.barh(
    pos_100d,
    [rast_100d_numpy, rast_10d_numpy, rast_1d_numpy],
    bar_width,
    alpha=opacity,
    color="#2ca02c",
    label="numpy",
)

# Add axis labels and title
ax.set_xlabel("μs")
ax.set_title("RBF benchmark")

# Add tick labels
ax.set_yticks(
    [
        np.mean(pos_100d) + 0.75,
        np.mean(pos_100d) + -0.25,
        np.mean(pos_100d) - 1.25,
    ]
)
ax.set_yticklabels(["1D", "10D", "100D"], color="black")

# Add legend
plt.legend()

# plt.style.use("dark_background")

# Display the plot
plt.savefig("rbf_bench_bar.png")
plt.show()
