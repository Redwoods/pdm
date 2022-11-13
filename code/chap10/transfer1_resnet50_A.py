from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
model.summary()

len(model.layers)

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))	# 영상 크기를 변경하고 적재한다.
x = image.img_to_array(img)	# 영상을 넘파이 배열로 변환한다. 
x = np.expand_dims(x, axis=0)	# 차원을 하나 늘인다. 배치 크기가 필요하다. 
x = preprocess_input(x)	# ResNet50이 요구하는 전처리를 한다. 

print(x.shape)

preds = model.predict(x)
print('예측:', decode_predictions(preds, top=3)[0])
