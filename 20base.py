# sh, bash, ksh, or zsh
# Tensorflow, Kereas, Anaconda, Virtualenv NEEDED!

cd /home/chenzhiyu
source ./venv/bin/activate
wget -nc https://media.nature.com/original/nature-assets/nbt/journal/v37/n1/extref/nbt.4317-S3.txt

python
import tensorflow as tf
import keras
import numpy as np
from random import shuffle

# SEQ=gRNA序列, SCA=Scaffold[Improved/Conventional]

with open('nbt.4317-S3.txt', 'r') as file:
    lines=file.readlines()[1:]
    shuffle(lines)
    data_n = len(lines)
    SEQ = np.zeros((data_n, 20, 4), dtype=int)
    SCA = np.zeros((data_n, 1), dtype=int)
    
    for l in range(0,data_n):
        data = lines[l].split()
        seq = data[1]
        for i in range(20):
            if seq[i] == "A":
                SEQ[l, i, 0] = 1
            elif seq[i] == "C":
                SEQ[l, i, 1] = 1
            elif seq[i] == "G":
                SEQ[l, i, 2] = 1
            elif seq[i] == "T":
                SEQ[l, i, 3] = 1
        sca = data[3]
        if sca == "Improved":
            SCA[l,0] = 1
        elif sca == "Conventional":
            SCA[l,0] = 0


'''
顺序 （请删除 shuffle(lines))
'''     

# 随机

train_data = SEQ[8194:]
train_labels = SCA[8194:]
test_data = SEQ[:8194]
test_labels = SCA[:8194]
x_val = train_data[0:8194]
partial_x_train = train_data[8194:]
y_val = train_labels[0:8194]
partial_y_train = train_labels[8194:]


# keras C1D卷积

model = keras.Sequential()
model.add(keras.layers.Convolution1D(filters=64, kernel_size=3, input_shape=(20,4), padding='same'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, epochs=128, batch_size=256, validation_data=(x_val, y_val), verbose=1)


# 测试

results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
history_dict.keys()

# 参数的图像表示

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, '-', label='Training loss', color='blue')
plt.plot(epochs, val_loss, '--', label='Validation loss' , color='red')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() 
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, '-', label='Training acc', color='blue')
plt.plot(epochs, val_acc, '--', label='Validation acc', color='red')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
