# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model

# import numpy as np

# class FeatureExtractor:
#     def __init__(self):
#         base_model = VGG16(weights="imagenet")
#         self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

#     def extractor(self, img):
#         img = img.resize((224,224)).convert("RGB")
#         x = image.img_to_array(img)  #to np array
#         x = np.expand_dims(x, axis=0) 
#         x = preprocess_input(x) #subtract avg pixel value
#         feature = self.model.predict(x)[0]
#         return feature / np.linalg.norm(feature)



from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize