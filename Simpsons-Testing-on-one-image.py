from keras.preprocessing import image
import matplotlib.pyplot as plt
img = image.load_img(r"C:\Python37\Projects\ALL ML-DL-DS Projects from Udemy and other Sources\DeepLearningCV\16. Design Your Own CNN - LittleVGG\simpsons\validation\ned_flanders\ned_flanders_28.jpg",target_size=(32,32))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model(r'C:\Python37\Projects\ALL ML-DL-DS Projects from Udemy and other Sources\DeepLearningCV\16. Design Your Own CNN - LittleVGG/simpsons_little_vgg.h5')
output = saved_model.predict_classes(img)[0]
#print(my_dirs[output])
plt.title(my_dirs[output])
plt.show()