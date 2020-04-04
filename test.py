from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img

def run_example():
	img = load_image('sample_image.png')
	model = load_model('final_model.h5')
	digit = model.predict_classes(img)
	print(digit[0])

run_example()