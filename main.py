import re

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


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


languages = ['english', 'germany', 'spanish', 'french', 'polish']
training_data = []
training_labels = []
test_data = []
test_labels = []

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

# Testowanie modelu na danych testowych
predictions_encoded = model.predict(test_data)
predictions = label_encoder.inverse_transform(predictions_encoded)
accuracy = accuracy_score(test_labels, predictions)
print('Accuracy:', accuracy)

# Uruchomienie predykcji dla użytkownika
end = False
print('To exit program press enter')
while not end:
    text = input(f"Enter test text: ")
    to_predict = prepare_user_data(text)

    if len(text) > 0:
        prediction_encoded = model.predict([to_predict])
        prediction = label_encoder.inverse_transform(prediction_encoded)
        print('Prediction:', prediction)
    else:
        end = True
