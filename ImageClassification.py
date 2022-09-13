from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import numpy as np
import pandas as pd
import joblib


def img_to_csv():
    column_names = list()
    column_names.append('label')
    for i in range(2500):
        column_names.append(str(i))

    data_frame = pd.DataFrame(columns=column_names)

    num_images = 0

    for i in range(0, 726):
        if (i < 10):
            img = Image.open("./data/airplane/airplane_000"+str(i)+".jpg")
            print("Added: " + "./data/airplane/airplane_000"+str(i)+".jpg")
        elif(i >= 10 and i < 100):
            img = Image.open("./data/airplane/airplane_00"+str(i)+".jpg")
            print("Added: " + "./data/airplane/airplane_00"+str(i)+".jpg")
        elif(i >= 100 and i < 1000):
            img = Image.open("./data/airplane/airplane_0"+str(i)+".jpg")
            print("Added: " + "./data/airplane/airplane_0"+str(i)+".jpg")

        img = img.resize((50, 50), Image.Resampling.NEAREST)
        img.load()

        img_data = np.asarray(img, dtype="int32")

        data = []
        data.append('airplane')
        for y in range(50):
            for x in range(50):
                data.append(img_data[x][y])
        data_frame.loc[num_images] = data
        num_images += 1

    for i in range(0, 967):
        if (i < 10):
            img = Image.open("./data/car/car_000"+str(i)+".jpg")
            print("Added: " + "./data/car/car_000"+str(i)+".jpg")
        elif(i >= 10 and i < 100):
            img = Image.open("./data/car/car_00"+str(i)+".jpg")
            print("Added: " + "./data/car/car_00"+str(i)+".jpg")
        elif(i >= 100 and i < 1000):
            img = Image.open("./data/car/car_0"+str(i)+".jpg")
            print("Added: " + "./data/car/car_0"+str(i)+".jpg")

        img = img.resize((50, 50), Image.Resampling.NEAREST)
        img.load()

        img_data = np.asarray(img, dtype="int32")

        data = []
        data.append('car')
        for y in range(50):
            for x in range(50):
                data.append(img_data[x][y])
        data_frame.loc[num_images] = data

        num_images += 1

    data_frame.to_csv("data.csv", index=False)


def create_model():
    df = pd.read_csv('data.csv')
    X_train = []
    y_train = []
    X = df.drop(columns=['label'])
    y = df['label']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    img = Image.open("C:/Users/Kirill/Downloads/airplane.jpg")
    img = img.resize((50, 50), Image.Resampling.NEAREST)
    img.load()

    img_data = np.asarray(img, dtype="int32")

    data = []
    for y in range(50):
        for x in range(50):
            data.append(img_data[x][y])
    predictions = model.predict(data)
    print(predictions)


create_model()
