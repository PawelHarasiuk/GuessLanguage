import os
import re
import sqlite3
import string
import tkinter as tk
import tkinter.ttk as ttk

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

root = tk.Tk()
root.title("Predykcja Języka")
root.geometry("800x600")


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
    label_encoder_rebuild = LabelEncoder()
    training_labels_encoded_rebuild = label_encoder_rebuild.fit_transform(training_labels)

    model = MLPClassifier(hidden_layer_sizes=(len(training_data),), max_iter=2000, random_state=42)

    model.fit(training_data, training_labels_encoded_rebuild)
    rebuild_text.config(text="Model rebuilt.")


def load_data():
    db = connection()
    cursor = db.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_data'")
    table_exists = cursor.fetchone()

    if table_exists:
        cursor.execute("SELECT * FROM training_data")
        data = cursor.fetchall()
    else:
        cursor.execute("CREATE TABLE training_data (label TEXT, data TEXT)")
        db.commit()
        data = []

    cursor.close()
    db.close()
    return data


def load_button_command():
    for item in treeview.get_children():
        treeview.delete(item)
    for lang, content in zip(training_labels, training_data):
        content_dict = {column: value for column, value in zip(columns, content)}
        treeview.insert("", "end", values=(lang, *content_dict.values()))


def connection():
    conn = sqlite3.connect('training_data.db')
    return conn


def visualize_data():
    model_weights = model.coefs_[0]
    languages = label_encoder.classes_

    for i, language in enumerate(languages):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'Weights - {language}')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Letter')

        x_labels = list(string.ascii_lowercase)
        weights = model_weights[i][:26]

        ax.bar(x_labels, weights, alpha=0.5)
        ax.grid(axis='y', linestyle='-')
        ax.grid(axis='x', linestyle='-')
        plt.tight_layout()
        plt.show()


def add_button_command():
    text = input_text.get("1.0", "end-1c")
    language = language_var.get()

    if len(text) > 0 and language:
        formatted_data = prepare_user_data(text)

        training_data.append(formatted_data)
        training_labels.append(language)

        db = connection()
        cursor = db.cursor()
        cursor.execute("INSERT INTO training_data (label, data) VALUES (?, ?)",
                       (language, str(formatted_data)))
        db.commit()
        cursor.close()
        db.close()

        input_text.delete("1.0", tk.END)

        save_text.config(text="Record added successfully.")
        load_button_command()
    else:
        save_text.config(text="Error: Empty text or no language selected.")


languages = os.listdir("data/training")
training_data = []
training_labels = []
test_data = []
test_labels = []
columns = [chr(i) for i in range(97, 123)]
for l in languages:
    for i in range(10):
        training_data.append(read_file(f'data/training/{l}/{i}.txt'))
        training_labels.append(l)
        test_data.append(read_file(f'data/test/{l}/{i}.txt'))
        test_labels.append(l)

label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
test_labels_encoded = label_encoder.transform(test_labels)

input_label = ttk.Label(root, text="Enter test text:")
input_label.pack()

input_text = tk.Text(root, height=10)
input_text.pack()

language_label = ttk.Label(root, text="Select language:")
language_label.pack()

language_var = tk.StringVar()
language_combobox = ttk.Combobox(root, textvariable=language_var, state="readonly")
language_combobox['values'] = languages
language_combobox.pack()

add_button = ttk.Button(root, text="Add", command=add_button_command)
add_button.pack()

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

save_text = ttk.Label(root, text="")
save_text.pack()

load_button = ttk.Button(root, text="Load Data", command=load_button_command)
load_button.pack()

visualize_button = ttk.Button(root, text="Visualize Data", command=visualize_data)
visualize_button.pack()

treeview_frame = ttk.Frame(root)
treeview_frame.pack(pady=20)

treeview = ttk.Treeview(treeview_frame)
treeview["columns"] = ["label"] + columns
for column in ["label"] + columns:
    treeview.column(column, anchor=tk.CENTER, width=100)
    treeview.heading(column, text=column)

data = load_data()
for row in data:
    treeview.insert("", "end", values=row)

treeview.pack()
root.mainloop()
