import logging
import pickle
import time
import traceback

import dill as dill
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from kan import KAN, ckpt

start_time = time.time()
try:
    # Laste inn datasettet
    df = pd.read_csv('login_data/login_data.csv')

    # Konvertere tid fra minutter siden midnatt til et tidspunkt i løpet av dagen
    df['hour'] = df['time'] // 60

    # Konverter 'date' til datetime
    df['date'] = pd.to_datetime(df['date'])

    # Få ukedagen som et tall
    df['weekday'] = df['date'].dt.dayofweek

    # Konvertere kategoriske data til numeriske verdier
    df['weekday'] = df['weekday'].astype('category').cat.codes
    df['geography'] = df['geography'].astype('category').cat.codes
    df['service'] = df['service'].astype('category').cat.codes
    df['user'] = df['user'].astype('category').cat.codes

    # Velge relevante kolonner for KAN
    data_for_kan = df[['weekday', 'geography', 'service', 'user']]

    # Konvertere til numpy arrays
    X = data_for_kan.values
    y = np.zeros(X.shape[0])  # Placeholder for målvariabel, kan brukes for unsupervised læring

    # Konvertere til PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Splitte datasettet i trenings- og testsett
    train_data, test_data, train_target, test_target = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Opprett data loaders (valgfritt, hvis du vil batch og shuffle data)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_target), batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_target), batch_size=1, shuffle=False)

    # Samle alle data i en enkelt tensor på den spesifiserte enheten
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    logging.info(f"Device: {device}. cuda equals gpu")

    train_inputs = torch.empty(0, 4, device=device)
    train_labels = torch.empty(0, dtype=torch.long, device=device)
    test_inputs = torch.empty(0, 4, device=device)
    test_labels = torch.empty(0, dtype=torch.long, device=device)

    for data, labels in train_loader:
        train_inputs = torch.cat((train_inputs, data.to(device)), dim=0)
        train_labels = torch.cat((train_labels, labels.to(device)), dim=0)

    for data, labels in test_loader:
        test_inputs = torch.cat((test_inputs, data.to(device)), dim=0)
        test_labels = torch.cat((test_labels, labels.to(device)), dim=0)

    dataset = {
        'train_input': train_inputs,
        'test_input': test_inputs,
        'train_label': train_labels,
        'test_label': test_labels
    }


    image_folder = 'video_img'
    # Initialisere KAN med nødvendige parametere
    model = KAN(width=[4, 5, 3], grid=5, k=3, seed=0, device=device)

    # Definere trenings- og testfunksjoner for nøyaktighet
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

    # Trene modellen
    results = model.train(dataset, opt="Adam", device=device, metrics=(train_acc, test_acc),
                          loss_fn=torch.nn.CrossEntropyLoss(), steps=100, lamb=0.01, lamb_entropy=10., save_fig=True, img_folder=image_folder)


    print(f"Train accuracy: {results['train_acc'][-1]}, Test accuracy: {results['test_acc'][-1]}")

    ckpt.saveckpt(model)
    # Use Pickle to save the model
    dill.dump(model, open('./validate_login.model', 'wb'))

    formula = model.symbolic_formula()

    print(f"Model formula: {formula}")
    print(f"Model index: {formula[0][0]}")


# model.auto_save   #.save('login_model.joblib')
except Exception as e:
    print(f"En feil oppstod: {e}")
    traceback.print_exc()


finally:
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Koden kjørte på {execution_time} sekunder.")
