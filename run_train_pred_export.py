from ultralytics import YOLO
# Create a new YOLO model from scratch
# model = YOLO('yolov8n.pt')
import cv2
# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/misa/PycharmProjects/ultralytics/runs/train5/weights/best.pt')

# source = '/home/misa/PycharmProjects/PaddleSeg/data/ekyc_doc_seg1234/testset1000/images'
# #source = '/home/misa/PycharmProjects/MISA.eKYC2/data/esign_ekyc_data/testset_1000/images/f72aab1d-cccd_fd3bc14259fa44e989e32b70e509e398637897872537590866.jpg'
# results = model.predict(source, save=True, imgsz=640, conf=0.5)
# img = cv2.imread(source)
# for result in results:
#     boxes = result.boxes.numpy().xyxy
#     for box in boxes:
#         cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0),3)
#
# cv2.imwrite('res.jpg', img)
# cv2.waitKey(0)
# Train the model using the 'coco128.yaml' dataset for 3 epochs

# Evaluate the model's performance on the validation set
# results = model.val(data='ekyc_doc_det1234.yaml')

# Export the model to ONNX format
success = model.export(format='onnx', imgsz=(640,640),opset=15)