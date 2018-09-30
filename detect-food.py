import sys
import io
import os

from google.cloud import vision
from google.cloud.vision import types

client = vision.ImageAnnotatorClient()

food_list = ["vegetables", "salmon", "chicken", "turkey", "salad", "fruits", "bread",
			"rice", "pork", "green beans", "strawberries", "broccoli", "corn", "pie", 
			"mashed potatoes", "salami", "kiwi", "chicken nuggets", "beef"
			 ]

argimg = sys.argv[1]

fname = os.path.join(
	os.path.dirname(__file__),
	argimg)

with io.open(fname, 'rb') as image_file:
	content = image_file.read()

img = types.Image(content=content)

response_labels = client.label_detection(image=img)
labels = response_labels.label_annotations


recognized_food = "meat"
for l in labels:
	for food in food_list:
		if l.description in food:
			recognized_food = food
			break
 
print('Recognized food', recognized_food)
print('Labels: ')
for l in labels:
	print(l.description) 

sys.exit(recognized_food)