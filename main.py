import os
import re
import sqlite3
import tkinter as tk
import tkinter.ttk as ttk

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Tworzenie okna głównego
root = tk.Tk()
root.title("Predykcja Języka")
root.geometry("800x600")


# Funkcje pomocnicze
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().lower()
    clean_text = re.sub(r'[^a-zA-Z]+', '', text)
    letter_count = {}
    for letter in clean_text:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1
    letter_ratio = []
    letters = [chr(i) for i in range(97, 123)]
    for letter in letters:
        if letter in letter_count and letter in letters:
            ratio = letter_count[letter] / len(clean_text)
            letter_ratio.append(ratio)
        elif letter in letters:
            letter_ratio.append(0)
    return letter_ratio


def prepare_user_data(text):
    clean_text = re.sub(r'[^a-zA-Z]+', '', text.lower())
    letter_count = {}
    for letter in clean_text:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1
    letter_ratio = []
    letters = [chr(i) for i in range(97, 123)]
    for letter in letters:
        if letter in letter_count and letter in letters:
            ratio = letter_count[letter] / len(clean_text)
            letter_ratio.append(ratio)
        elif letter in letters:
            letter_ratio.append(0)
    return letter_ratio


def predict_language():
    text = input_text.get("1.0", "end-1c")
    to_predict = prepare_user_data(text)
    if len(text) > 0:
        # Predykcja dla użytkownika
        prediction_encoded = model.predict([to_predict])
        prediction = label_encoder.inverse_transform(prediction_encoded)
        prediction_text.config(text="Prediction: " + str(prediction))
    else:
        prediction_text.config(text="")


def evaluate_model():
    predictions_encoded = model.predict(test_data)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    accuracy = accuracy_score(test_labels, predictions)
    accuracy_text.config(text="Accuracy: " + str(accuracy))


def rebuild_model():
    global model
    # Przygotowanie i trenowanie modelu
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)
    model.fit(training_data, training_labels_encoded)
    rebuild_text.config(text="Model rebuilt.")


def save_data():
    db = connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM training_data")
    for data, label in zip(training_data, training_labels):
        cursor.execute("INSERT INTO training_data (label, data) VALUES (?, ?);", (label, data))
    db.commit()
    cursor.close()
    db.close()
    save_text.config(text="Data saved.")


def load_data():
    db = connection()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM training_data")
    data = cursor.fetchall()
    cursor.close()
    db.close()

    treeview.delete(*treeview.get_children())  # Clear existing data in the treeview

    for row in data:
        lang = row[0]
        content = row[1]
        row_data = content.split(", ")
        values = [item.split(" = ") for item in row_data]
        values = {item[0]: item[1] for item in values}
        row_values = [values.get(chr(97 + i), '0') for i in range(26)]
        treeview.insert("", "end", values=(lang, *row_values))


def connection():
    conn = sqlite3.connect('training_data.db')
    return conn


# Wczytanie danych treningowych
languages = os.listdir("data/training")
training_data = []
training_labels = []
test_data = []
test_labels = []
columns = ['data']
for l in languages:
    for i in range(10):
        training_data.append(read_file(f'data/training/{l}/{i}.txt'))
        training_labels.append(l)
        test_data.append(read_file(f'data/test/{l}/{i}.txt'))
        test_labels.append(l)

# Przekształcenie etykiet na wartości liczbowe
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Przygotowanie i trenowanie modelu
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)
model.fit(training_data, training_labels_encoded)

# Tworzenie interfejsu użytkownika
input_label = ttk.Label(root, text="Enter test text:")
input_label.pack()

input_text = tk.Text(root, height=10)
input_text.pack()

predict_button = ttk.Button(root, text="Predict", command=predict_language)
predict_button.pack()

prediction_text = ttk.Label(root, text="")
prediction_text.pack()

evaluate_button = ttk.Button(root, text="Evaluate Model", command=evaluate_model)
evaluate_button.pack()

accuracy_text = ttk.Label(root, text="")
accuracy_text.pack()

rebuild_button = ttk.Button(root, text="Rebuild Model", command=rebuild_model)
rebuild_button.pack()

rebuild_text = ttk.Label(root, text="")
rebuild_text.pack()

save_button = ttk.Button(root, text="Save Data", command=save_data)
save_button.pack()

save_text = ttk.Label(root, text="")
save_text.pack()

load_button = ttk.Button(root, text="Load Data", command=load_data)
load_button.pack()

treeview_frame = ttk.Frame(root)
treeview_frame.pack(pady=20)

treeview = ttk.Treeview(treeview_frame)
treeview["columns"] = ["language"] + [chr(97 + i) for i in range(26)]
for column in ["language"] + [chr(97 + i) for i in range(26)]:
    treeview.column(column, anchor=tk.CENTER, width=50)
    treeview.heading(column, text=column.upper())

load_data()

treeview.pack()

root.mainloop()
