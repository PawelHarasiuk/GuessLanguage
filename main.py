import random
import re


def read_file(file_path, language):
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
    litery = [chr(i) for i in range(97, 123)]

    for letter in litery:
        if letter in letter_count and letter in litery:
            ratio = letter_count[letter] / len(clean_text)
            letter_ratio.append(ratio)
        elif letter in litery:
            letter_ratio.append(0)

    letter_ratio.append(language)
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
    litery = [chr(i) for i in range(97, 123)]

    for letter in litery:
        if letter in letter_count and letter in litery:
            ratio = letter_count[letter] / len(clean_text)
            letter_ratio.append(ratio)
        elif letter in litery:
            letter_ratio.append(0)

    return letter_ratio


def epoki(expected_ratio):
    ratio = 0.0
    count = 0
    while ratio < expected_ratio:
        for language in languages:
            train(training_data, language)
        ratio = guess_all_test(False)
        count += 1

    print('liczba epok: ', count)


def train(trainig_data, language):
    for wektor_wejsciowy in trainig_data:
        net = 0
        length = len(wektor_wejsciowy) - 1 if len(wektor_wejsciowy) <= len(weights[language]) else len(
            weights[language])
        for i in range(length):
            net += float(wektor_wejsciowy[i]) * weights[language][i]

        net += weights[language][-1] * 1

        wynik = 1 if net >= 0.0 else 0
        oczekiwana_wartosc = 0
        if wektor_wejsciowy[-1] == language:
            oczekiwana_wartosc = 1

        if oczekiwana_wartosc != wynik:
            scale_weight(wektor_wejsciowy, wynik, oczekiwana_wartosc, language)


def scale_weight(wektor_wejsciowy, wynik, oczekiwana_wartosc, szukana_wartosc):
    length = len(wektor_wejsciowy) - 1 if len(wektor_wejsciowy) <= len(weights[szukana_wartosc]) else len(
        weights[szukana_wartosc])
    for i in range(length):
        weights[szukana_wartosc][i] += (oczekiwana_wartosc - wynik) * float(wektor_wejsciowy[i]) * alfa

    weights[szukana_wartosc][-1] += (oczekiwana_wartosc - wynik) * 1


def guess_all_test(enable_show):
    i = 0
    count = 0
    for wektor_wejsciowy in test_data:
        my_dict = predicate_test(wektor_wejsciowy)
        max_key = max(my_dict, key=my_dict.get)
        if max_key == test_data[i][-1]:
            count += 1
        if enable_show:
            print(my_dict)
            print('Prediction:', max_key)
            print('Real Value:', test_data[i][-1], '\n')
        i += 1

    if enable_show:
        print(count, '/', len(test_data), '=', count / len(test_data))

    return count / len(test_data)


def predicate_test(wektor_wejsciowy):
    wyniki_net = {}
    for language in languages:
        net = 0
        length = len(wektor_wejsciowy) - 1 if len(wektor_wejsciowy) <= len(weights[language]) else len(
            weights[language])
        for i in range(length):
            net += float(wektor_wejsciowy[i]) * weights[language][i]
        net += weights[language][-1] * 1
        wyniki_net[language] = net

    return wyniki_net


def predicate_user(wektor_wejsciowy):
    wyniki_net = {}

    for language in languages:
        net = 0
        length = len(wektor_wejsciowy) if len(wektor_wejsciowy) <= len(weights[language]) else len(weights[language])

        for i in range(length):
            net += float(wektor_wejsciowy[i]) * weights[language][i]

        net += weights[language][-1] * 1
        wyniki_net[language] = net

    my_dict = predicate_test(wektor_wejsciowy)
    max_key = max(my_dict, key=my_dict.get)

    print(my_dict, max_key)
    print('Prediction:', max_key)


def run_user():
    end = False
    print('To exit program press enter')
    while not end:
        text = input(f"Enter test text: ")
        to_predict = prepare_user_data(text)

        if len(text) > 0:
            predicate_user(to_predict)
        else:
            end = True


languages = ['english', 'germany', 'spanish', 'french', 'polish']
weights = {}
alfa = 0.3
training_data = []
test_data = []

for l in languages:
    weights[l] = []

for l in languages:
    for i in range(10):
        training_data.append(read_file(f'data/training/{l}/{i}.txt', f'{l}'))
        test_data.append(read_file(f'data/test/{l}/{i}.txt', f'{l}'))

for l in languages:
    new_weight = []

    for i in range(len(training_data[0])):
        new_weight.append(random.uniform(-1, 1))

    weights[l] = new_weight

epoki(0.90)
guess_all_test(True)
print(100*'=')
run_user()
