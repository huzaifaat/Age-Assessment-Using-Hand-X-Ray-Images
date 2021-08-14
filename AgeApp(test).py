import tkinter as tk
import matplotlib as plt
import cv2
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile

import pathlib
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from tkinter import ttk
import cv2
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten
from keras.metrics import mean_absolute_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.metrics import accuracy
PATH=""
file=""
file_name="t1.png"

root = tk.Tk()
pr=0
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=3)

#logo
logo = Image.open('logo.png')
logo = ImageTk.PhotoImage(logo)
original_img= tk.Label(image=logo)
original_img.image = logo
original_img.grid(column=1, row=0)


#instructions
instructions = tk.Label(root, text="AGE Estimation Using Hand X-Ray", font="Monaco")
instructions.grid(columnspan=5, column=0, row=1)
instructions = tk.Label(root, text="Upload Image And get Results", font="Monaco")
instructions.grid(columnspan=5, column=0, row=2)
# def open_file():
#     browse_text.set("...Loading...")
#     file = askopenfile(parent=root, mode='rb', title="Choose a file", filetypes=[("PNG file", ".png")])
#     if file:
#         read_img = prepare(file)
#         image_content = plt.imshow(read_img)
#
#         #text box
#         image_box = tk.image(root, height=20, width=50, padx=15, pady=15)
#         image_box.insert(1.0, image_content)
#         image_box.tag_configure("center", justify="center")
#         image_box.tag_add("center", 1.0, "end")
#         image_box.grid(column=1, row=3)
#
#         browse_text.set("Browse")
def open_file():
    global file
    try:
        file = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=[('Images', ['*jpeg', '*png', '*jpg'])])
        # self.label_file = Label(self.left_frame, width=100, height=4, fg="blue", text="File Opened: " + file).grid(column = 1, row = 2)
        global PATH

        PATH = file
        #print(self.PATH)
        # copy2(self.PATH, "./input/t1.png")

        # Splits at /
        # selected_file = (self.PATH.split("/"))
        # self.file_pat.insert(0, "file opened :" + selected_file[-1])

        file = Image.open(PATH)

        file.save(pathlib.Path("tmp/" + file_name))

        # file.save(pathlib.Path("tmp/"+file_name))

        file = file.resize((256, 256))
        file = ImageTk.PhotoImage(file)
        original_img.configure(text="Loaded image", image=file, padx=250, pady=250)
        original_img.text = ""
        original_img.image = file

        # Add text in Entry box
    except Exception as e:
        messagebox.showerror("An error occured !", e)

browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=open_file, font="Raleway", bg="#0000FF", fg="white",
                       height=2, width=15)
def predict_age():

    img_size = 256

    def mae_in_months(x_p, y_p):
        '''function to return mae in months'''
        return mean_absolute_error((std_bone_age * x_p + mean_bone_age), (std_bone_age * y_p + mean_bone_age))

    model_1 = tf.keras.applications.xception.Xception(input_shape=(img_size, img_size, 3),
                                                      include_top=False,
                                                      weights='imagenet')
    model_1.trainable = True
    model_2 = Sequential()
    model_2.add(model_1)
    model_2.add(GlobalMaxPooling2D())
    model_2.add(Flatten())
    model_2.add(Dense(10, activation='relu'))
    model_2.add(Dense(1, activation='linear'))

    model_2.compile(loss='mse', optimizer='adam', metrics=[mae_in_months])

    model_2.load_weights('best_model.h5')


    img_size = 256
   # loading dataframes
    train_df = pd.read_csv(r'C:\Users\l1f16bscs0450\PycharmProjects\AgeCNN\kaggle\input\boneage-training-dataset.csv')
    test_df = pd.read_csv(r'C:\Users\l1f16bscs0450\PycharmProjects\AgeCNN\kaggle\input\boneage-test-dataset.csv')
    mean_bone_age = train_df['boneage'].mean()
    print('mean: ' + str(mean_bone_age))
    std_bone_age = train_df['boneage'].std()
    train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age) / (std_bone_age)

    print(train_df.head())

    # appending file extension to id column for both training and testing dataframes
    train_df['id'] = train_df['id'].apply(lambda x: str(x) + '.png')
    test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x) + '.png')
    # splitting train dataframe into traininng and validation dataframes
    df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=0)
    train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_data_generator.flow_from_dataframe(
        dataframe=df_valid,
        directory=r'C:\Users\PycharmProjects\AgeCNN\kaggle\input\boneage-training-dataset\boneage-training-dataset',
        x_col='id',
        y_col='bone_age_z',
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode='other',
        flip_vertical=True,
        color_mode='rgb',
        target_size=(img_size, img_size))
    test_X, test_Y = next(val_data_generator.flow_from_dataframe(
        df_valid,
        directory=r'C:\Users\PycharmProjects\AgeCNN\kaggle\input\boneage-training-dataset\boneage-training-dataset',
        x_col='id',
        y_col='bone_age_z',
        target_size=(img_size, img_size),
        batch_size=2523,
        class_mode='other'
    ))

    def prepare(filepath):
        IMG_SIZE = 256  # 50 in txt-based
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = cv2.cvtColor(new_array, cv2.COLOR_GRAY2RGB)
        print("image_size", new_array.shape)

        return new_array.reshape(-1, 256, 256, 3)

    #
    pred = mean_bone_age + std_bone_age * (model_2.predict
                                           (prepare(r'C:\Users\PycharmProjects\AgeApp\tmp\t1.png'),
                                                           batch_size=1, verbose=True))
    test_months = mean_bone_age + std_bone_age * (test_Y)
    global pr
    pr = pred / 12
    pr = int(pr)
    print(pr)



predict_btn = tk.Button(root, text="Predict Age", command=predict_age, font="Raleway", bg="#0000FF", fg="white",
                        height=2, width=15)
pr_btn = tk.Button(root, text=pr , font="Raleway", bg="#0000FF", fg="white",
                        height=2, width=15)

browse_text.set("Browse")

browse_btn.grid(column=1, row=4)

predict_btn.grid(column=1, row=5)

pr_btn.grid(column=1, row=6)

canvas = tk.Canvas(root, width=600, height=50)
canvas.grid(columnspan=3)




#browse button


root.mainloop()
print('is this working')