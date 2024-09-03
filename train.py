from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
# convolutional neural network construction
model = Sequential()

model.add(Conv2D(8,kernel_size=(3,3),input_shape=(400,400,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])




# image preprocessing

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   rotation_range=20.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)
        
val_datagen = ImageDataGenerator(rescale=1./255)

# import images

train_image = train_datagen.flow_from_directory(r"D:\Pantech ai\basic pgm in keras & tensorflow\Hand sign recognition\dataset\train",
                                                target_size=(400,400),
                                                color_mode='rgb',
                                                classes = ['0','1','2','3','4','5','6','7','8','9'],
                                                class_mode='categorical')
val_image = val_datagen.flow_from_directory(r"D:\Pantech ai\basic pgm in keras & tensorflow\Hand sign recognition\dataset\val",
                                            target_size=(400,400),
                                            color_mode='rgb',
                                            class_mode='categorical',
                                            classes = ['0','1','2','3','4','5','6','7','8','9'])


callback_list = [
    EarlyStopping(monitor='val_loss',patience=10),
    ModelCheckpoint(filepath='model.h5',monitor='val_loss',save_best_only=True,verbose=1)]


model.fit_generator(train_image,
                    epochs = 5,
                    validation_data=val_image,
                    callbacks=callback_list)
model_json = model.to_json()

with open("model.json","w") as f:
    f.write(model_json)
