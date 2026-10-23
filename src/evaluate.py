import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")
loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
print("Test Loss:", loss, "Test Accuracy:", acc)

pred = model.predict(d["x_test"]).argmax(axis=1)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot()
plt.savefig("models/confusion_matrix.png")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)},
          open("metrics.json", "w"), indent=2)
