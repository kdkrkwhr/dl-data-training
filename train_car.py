import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json

from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# Model 생성
cnn = Sequential()

# 첫 번째 컨볼루션 및 풀링 레이어 추가
cnn.add(Conv2D(32, kernel_size = (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Dropout(0.2))

# 두 번째 컨볼루션 및 풀링 계층 추가
cnn.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Dropout(0.2))

# 세 번째 컨볼루션 및 풀링 계층 추가
cnn.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Dropout(0.2))

# 네 번째 컨볼루션 및 풀링 계층 추가
cnn.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Dropout(0.2))

#다섯 번째 컨볼루션 및 풀링 계층 추가
cnn.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Dropout(0.2))

# Conv2D Layer 와 MaxPool2D Layer는 2차원 자료를 다루지만 전달 하기위해선 1차원 자료로 변경 필요
# Flatten 은 2차원 Layer를 1차원으로 변경해주는 Layer
cnn.add(Flatten())

# 출력 레이어 추가
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 128, activation = 'relu'))

cnn.add(Dense(units = 3, activation = 'softmax'))

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# data 변환
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_data = train_datagen.flow_from_directory('./input/data/train_data',
                                              target_size = (128,128),
                                              batch_size = 32,
                                              class_mode = 'categorical')

test_data = test_datagen.flow_from_directory('./input/data/test_data',
                                              target_size = (128,128),
                                              batch_size = 32,
                                              class_mode = 'categorical')

history = cnn.fit_generator(train_data,
                            steps_per_epoch = 100,
                            epochs = 60,
                            validation_data = test_data,
                            validation_steps = 50)

vals = pd.DataFrame.from_dict(history.history)
vals = pd.concat([pd.Series(range(0, 30), name = 'epochs'), vals], axis = 1)
vals.head()

cnn.save('cdc.h5')

model_json = cnn.to_json()
with open("cdc.json", "w") as json_file:
    json_file.write(model_json)

hi = history.history
loss = hi['loss']
val_acc = hi['val_acc']
acc = hi['acc']

epo = range(1, len(acc) + 1)

plt.plot(epo, val_acc, 'bo', label = 'Validation acc')
plt.plot(epo, acc, 'b', label = 'Training acc')

plt.title('TCV')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()
