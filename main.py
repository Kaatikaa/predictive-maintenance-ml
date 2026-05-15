import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras

#loading csv into a dataframe
df = pd.read_csv('ai4i2020.csv')

#remove empty rows
new_df = df.dropna()

#remove duplicates
new_df = new_df.drop_duplicates()

#remove failure type columns
new_df = new_df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)

#remove ID columns
new_df = new_df.drop(['UDI', 'Product ID'], axis=1)

#one-hot encoding Type column
new_df = pd.get_dummies(new_df, columns=['Type'])

#select input(x) and target(y)
x = new_df.drop('Machine failure', axis=1)
y = new_df['Machine failure']

#split into training(70%) and test(30%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#scaling (standard)
scaler = StandardScaler()
scale_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
x_train[scale_columns] = scaler.fit_transform(x_train[scale_columns])
x_test[scale_columns] = scaler.transform(x_test[scale_columns])

#neurális háló definiálása
model = keras.models.Sequential(name = 'Modell')
model.add(keras.layers.Input(shape=(8,)))
model.add(keras.layers.Dense(16, activation='relu', name = 'hidden-layer1'))
model.add(keras.layers.Dense(8, activation='relu', name = 'hidden-layer2'))
model.add(keras.layers.Dense(1, activation='sigmoid', name = 'output'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#tanítás
model_history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, shuffle=True, verbose=0)
model_history.history

#kiértékelés teszt adatokon
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

#user input kérése
answer = input("Szeretne megadni saját adatokat? (y/n): ").lower()
if answer == "y":
    machine_type = input("Type (L, M, H): ").upper()
    air_temp = float(input("Air temperature [K]: "))
    process_temp = float(input("Process temperature [K]: "))
    rotational_speed = float(input("Rotational speed [rpm]: "))
    torque = float(input("Torque [Nm]: "))
    tool_wear = float(input("Tool wear [min]: "))

    type_L = 1 if machine_type == 'L' else 0
    type_M = 1 if machine_type == 'M' else 0
    type_H = 1 if machine_type == 'H' else 0

    
    user_df = pd.DataFrame([{
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type_H': type_H,
        'Type_L': type_L,
        'Type_M': type_M
    }])

    user_df[scale_columns] = scaler.transform(user_df[scale_columns])

    prediction = model.predict(user_df)

    probability = prediction[0][0] * 100

    print(f"Meghibásodás valószínűsége: {probability:.2f}%")
