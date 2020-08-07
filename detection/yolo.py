import cv2
import numpy as np 
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("/home/cc/nanonets_object_tracking/detection/yolov3.weights", "/home/cc/nanonets_object_tracking/detection/yolov3.cfg")
	classes = []
	with open("/home/cc/nanonets_object_tracking/detection/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = (detect[0] * width)
				center_y = (detect[1] * height)
				w = (detect[2] * width)
				h = (detect[3] * height)
				x = (center_x - w/2)
				y = (center_y - h/2)
				
				# center_x = int(detect[0] * width)
				# center_y = int(detect[1] * height)
				# w = int(detect[2] * width)
				# h = int(detect[3] * height)
				# x = int(center_x - w/2)
				# y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), color, 2)
			cv2.putText(img, label, (int(x), int(y - 5)), font, 1, color, 1)
#	cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video_path, out):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(fps)
	while True:
		#if cv2.waitKey(1) & 0xFF == ord('q'): break
		ret, frame = cap.read()
#		print(frame)
		if ret is False:
			frame_id+=1
			break	
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		#fps = cap.get(cv2.CAP_PROP_FPS)
		frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
		#frame_id = cv2.cvGetCaptureProperty(cap,1)
		det = {'frame_id':frame_id, 'boxes':boxes, 'confs':confs}
		#print(det['boxes'])
		#for i in range(len(boxes)):
		#	print(det['boxes'][i])
		#print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		for i in range(len(boxes)):
			out.write('%d, -1, '%frame_id)
			out.write('%.3f, %.3f, %.3f, %.3f, %.3f, '%(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],confs[i]))
			out.write('-1, -1, -1\n')
			#out.write('{0[frame_id]}, -1, {0[boxes]}, {0[confs]}, -1 -1 -1\n'.format(det))
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
		#key = cv2.waitKey(1)
		#if key == 27:
		#	break
	cap.release()
	print("start_video called")

if __name__ == '__main__':
	print("start of main")
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	vpath = os.getcwd()+args.video_path
	base = os.path.basename(vpath)	
	det_name = "det_"+os.path.splitext(base)[0]+".txt"
	out = open("./det/"+det_name, "wt")
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path, out)
		print("video_play done")
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
	print(det_name)
	print(type(out))
	cv2.destroyAllWindows()
