import os
import re
from io import BytesIO

import flask
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_file

app = Flask(__name__)

def extract_g_l1_sm_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            str_num=line.split("G_L1_sm:").split("G_L1_kong").replace(" ","")

            number = float(str_num.group(1))
            numbers.append(number)
    return numbers

def traverse_directory_and_extract_numbers(folder_path):
    all_numbers = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'loss_log.txt':
                file_path = os.path.join(root, file)
                numbers = extract_g_l1_sm_numbers_from_file(file_path)
                all_numbers[file_path] = numbers
    return all_numbers

def plot_graph(data):
    plt.figure(figsize=(10, 6))
    for file_path, numbers in data.items():
        plt.plot(range(len(numbers)), numbers, label=os.path.basename(file_path))

    plt.xlabel('Line Number')
    plt.ylabel('G_L1_sm Value')
    plt.title('G_L1_sm Values from loss_log.txt')
    plt.legend()
    plt.tight_layout()

@app.route('/')
def home():
    folder_path = '/hdd/share/quality/checkpoints'  # replace with the path to your folder
    numbers_by_file = traverse_directory_and_extract_numbers(folder_path)

    plot_graph(numbers_by_file)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
