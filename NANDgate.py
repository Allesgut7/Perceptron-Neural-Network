import tkinter as tk
from tkinter import messagebox
import numpy as np

# Fungsi-fungsi perceptron

def sigmoid(x): #mendefinisikan activation function menggunakan sigmoid
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x): #mendefinisikan turunan sigmoid
    return sigmoid(x) * (1 - sigmoid(x))

def update_weights(x, y, W, learning_rate): #mendefinisikan fungsi update bobot dengan menggunakan turunan sigmoid
    Z = np.dot(x, W)
    Y_hat = sigmoid(Z)
    error = y - Y_hat
    dEdW = error * derivative_sigmoid(Z) * x
    W += learning_rate * dEdW
    return W, error

# Fungsi untuk menampilkan GUI

def predict():
    x = np.array([1, float(entry_x1.get()), float(entry_x2.get())])
    Z = np.dot(x, weights)
    Y_pred = sigmoid(Z)
    prediction = 1 if Y_pred >= 0.5 else 0
    label_result.config(text="Hasil Prediksi: {}".format(prediction))
    label_sigmoid.config(text="Hasil Sigmoid: {:.4f}".format(Y_pred))

def train():
    global weights
    learning_rate = float(entry_learning_rate.get())
    iterations_before_update = int(entry_iterations_before_update.get())
    for _ in range(iterations_before_update):
        for i in range(X.shape[0]):
            x = np.array([1, X[i][0], X[i][1]])
            weights, error = update_weights(x, Y[i], weights, learning_rate)
            if error != 0:
                messagebox.showinfo("Info", "Terjadi Kesalahan, Dilakukan Updating Bobot")
                break
    predict()
    label_weights_x0.config(text="W0: {:.4f}".format(weights[0]))
    label_weights_x1.config(text="W1: {:.4f}".format(weights[1]))
    label_weights_x2.config(text="W2: {:.4f}".format(weights[2]))

# Inisialisasi bobot dan data latih

weights = np.random.rand(3) #bobot awaal dengan nilai random
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
Y = np.array([1, 1, 1, 0])

# Membuat GUI

root = tk.Tk()
root.title("NAND Gate 0200_Galih")

frame_inputs = tk.Frame(root)
frame_inputs.pack()

label_x0 = tk.Label(frame_inputs, text="Bias Unit (x0) :")
label_x0.grid(row=0, column=0)
label_bias = tk.Label(frame_inputs, text=" 1 ")
label_bias.grid(row=0, column=1)
label_weights_x0 = tk.Label(frame_inputs, text="W0: {:.4f}".format(weights[0]))
label_weights_x0.grid(row=0, column=2)

label_x1 = tk.Label(frame_inputs, text="Masukkan Input Pertama (x1):")
label_x1.grid(row=1, column=0)
entry_x1 = tk.Entry(frame_inputs)
entry_x1.grid(row=1, column=1)

label_weights_x1 = tk.Label(frame_inputs, text="W1: {:.4f}".format(weights[1]))
label_weights_x1.grid(row=1, column=2)

label_x2 = tk.Label(frame_inputs, text="Masukkan Input Kedua (x2):")
label_x2.grid(row=2, column=0)
entry_x2 = tk.Entry(frame_inputs)
entry_x2.grid(row=2, column=1)

label_weights_x2 = tk.Label(frame_inputs, text="W2: {:.4f}".format(weights[2]))
label_weights_x2.grid(row=2, column=2)

label_learning_rate = tk.Label(frame_inputs, text="Learning Rate:")
label_learning_rate.grid(row=3, column=0)
entry_learning_rate = tk.Entry(frame_inputs)
entry_learning_rate.grid(row=3, column=1)

label_iterations_before_update = tk.Label(frame_inputs, text="Iterasi Sebelum Update:")
label_iterations_before_update.grid(row=4, column=0)
entry_iterations_before_update = tk.Entry(frame_inputs)
entry_iterations_before_update.grid(row=4, column=1)

button_train = tk.Button(root, text="Latih", command=train)
button_train.pack()

label_result = tk.Label(root, text="Hasil Prediksi:")
label_result.pack()

label_sigmoid = tk.Label(root, text="Hasil Sigmoid:")
label_sigmoid.pack()

root.mainloop()
