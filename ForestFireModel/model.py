import pandas as pd
import numpy as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

train_dir='Data/Train_Data'
test_dir='Data/Test_Data'

from PIL import Image
import os

def check_images(directory):
    """
    Bu fonksiyon belirtilen klasördeki tüm görüntü dosyalarını kontrol eder.
    Eğer görüntü bozuksa, dosya adı yazdırılır ve bozuk dosyalar kaldırılabilir.
    """
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                img = Image.open(file_path)  # Görüntüyü açmayı dener
                img.verify()  # Görüntünün bozuk olup olmadığını kontrol eder
            except (IOError, SyntaxError) as e:
                print(f'Bozuk dosya tespit edildi: {file_path}')
                os.remove(file_path)  # Bozuk dosyayı silebilirsiniz (isteğe bağlı)

# Eğitim ve test klasörlerini kontrol edin
check_images(train_dir)
check_images(test_dir)

# 2. Veri Analizi ve Görselleştirme
def plot_sample_images(generator):
    class_labels = list(generator.class_indices.keys())
    for i in range(0, len(class_labels)):
        img_path = generator.filepaths[i]
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"Label: {class_labels[i]}")
        plt.show()

# 3. Veri Ön İşleme
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# 4. Model Kurulumu (ResNet-50 Transfer Learning)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (Fire or Non-Fire)

# 5. Model Derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 6. Model Callbacks (Early Stopping, Checkpoint)
checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=10)

# 7. Model Eğitimi
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    callbacks=[checkpoint, early_stopping])

# 8. Eğitim Performansını Görselleştirme
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# 9. Test Seti ile Model Performansı
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# 10. Modelin Kaydedilmesi
model.save('final_fire_detection_model.h5')
