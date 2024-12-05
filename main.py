import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_files = os.listdir("data")
en_files = [f"data/{file}" for file in data_files if "en" in file]
gm_files = [f"data/{file}" for file in data_files if "gm" in file]

en_data = pd.concat([pd.read_csv(file) for file in en_files], ignore_index=True)
gm_data = pd.concat([pd.read_csv(file) for file in gm_files], ignore_index=True)

en_data["date"] = pd.to_datetime(en_data[["year", "month", "day"]].assign(minute=0))
gm_data["date"] = pd.to_datetime(gm_data[["Year", "Month", "Day"]].assign(minute=0))

en_data.columns = map(str.lower, en_data.columns)
gm_data.columns = map(str.lower, gm_data.columns)

merged_data = pd.merge(
    en_data,
    gm_data,
    on=["warehouse", "date", "traffic_stream"],
    how="inner"
)

grouped_data = merged_data.groupby(["warehouse", "traffic_stream", "date"]).agg(
    klk_en_mean=("klk_en", "mean"),
    total_count_mean=("total_count", "mean")
).reset_index()

X = grouped_data["klk_en_mean"].values
y = grouped_data["total_count_mean"].values

X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
a = numerator / denominator
b = y_mean - a * X_mean

print(f"Рівняння регресії: total_count = {a:.2f} * klk_en + {b:.2f}")

plt.scatter(X, y, label="Дані")
plt.plot(X, a * X + b, color="red", label="Лінія регресії")
plt.xlabel("Середня кількість ЕН")
plt.ylabel("Середня кількість ГП")
plt.legend()
plt.show()

X_transformed = np.log(X + 1)
coefficients = np.polyfit(X_transformed, y, 1)
log_model = np.poly1d(coefficients)

plt.scatter(X_transformed, y, label="Трансформовані дані")
plt.plot(X_transformed, log_model(X_transformed), color="purple", label="Логарифмічна модель")
plt.xlabel("log(Середня кількість ЕН)")
plt.ylabel("Середня кількість ГП")
plt.legend()
plt.title("Логарифмічна регресія")
plt.show()

historical_en_data = en_data[en_data["date"] < pd.Timestamp("2021-08-01")]
historical_en_data["total_count_pred"] = a * historical_en_data["klk_en"] + b

avg_sum = gm_data["total_sum"].sum() / gm_data["total_count"].sum()
historical_en_data["total_sum_pred"] = historical_en_data["total_count_pred"] * avg_sum

historical_en_data.to_csv("predicted_gm_data.csv", index=False)
