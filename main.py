from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Flatten # type: ignore
from keras.layers import Conv2D, MaxPooling2D # type: ignore
import os

train_data_dir = 'data_fer2013/train'
validation_data_dir = 'data_fer2013/test/'


train_datagen = ImageDataGenerator(
					rescale = 1./255,
					rotation_range = 30,
					shear_range = 0.3,
					zoom_range = 0.3,
					horizontal_flip = True,
					fill_mode = 'nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode = 'grayscale',
					target_size = (48, 48),
					batch_size = 32,
					class_mode = 'categorical',
					shuffle = True)
validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode = 'grayscale',
							target_size = (48, 48),
							batch_size = 32,
							class_mode = 'categorical',
							shuffle = True)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
img, label = train_generator.__next__()


# Creación e impresión del modelo (CNN)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Contar imagenes de entrenamiento y prueba

train_path = train_data_dir
test_path = validation_data_dir

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)
print(num_train_imgs)
print(num_test_imgs)

# Entrenamiento del modelo

epocas = 100

history = model.fit(train_generator,
                    steps_per_epoch = num_train_imgs//32,
                    epochs = epocas,
                    validation_data = validation_generator,
                    validation_steps = num_test_imgs//32)

model.save('CNN_100E.keras')
# model.save(f"CNN_{epocas}E.keras")