from ultralytics import YOLO
import cv2, os, random
from common import resize_normalize
# Load a pretrained YOLO model (recommended for training)
model = YOLO('/home/misa/PycharmProjects/ultralytics/runs/detect/train9/weights/best.onnx')

def crop_source(img_dir, anno_dir, extend_range = 0.1, resize_width = 800):
    img_crop_dir = img_dir+'_crop3'
    anno_crop_dir = anno_dir+'_crop3'
    if not os.path.exists(img_crop_dir): os.makedirs(img_crop_dir)
    if not os.path.exists(anno_crop_dir): os.makedirs(anno_crop_dir)
    results = model.predict(img_dir, save=False, imgsz=640, conf=0.5)
    count = 0
    for n, result in enumerate(results):
        # if n>2000: continue
        img_path = result.path
        print(n, img_path)
        base_name = os.path.basename(img_path).split('.')[0]
        anno_path = os.path.join(anno_dir, base_name +'.png')
        img = cv2.imread(img_path)
        anno = cv2.imread(anno_path)
        img_w, img_h = img.shape[1],img.shape[0]
        boxes = result.boxes.numpy().xyxy
        extend = random.uniform(0.0, 0.1)
        if len(boxes)>0:
            for idx, box in enumerate(boxes):
                w,h = box[2]-box[0],box[3]-box[1]
                extend_x, extend_y = extend*w, extend*h
                crop_x1 = int(max(0, box[0] - extend_x))
                crop_y1 = int(max(0, box[1] - extend_y))
                crop_x2 = int(min(img_w-1, box[2] + extend_x))
                crop_y2 = int(min(img_h-1, box[3] + extend_y))
                #cv2.rectangle(img, (crop_x1,crop_y1), (crop_x2,crop_y2), (255,0,0),3)
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_anno =anno[crop_y1:crop_y2, crop_x1:crop_x2]
                if resize_width is not None:
                    crop_img, ratio = resize_normalize(crop_img, resize_width, interpolate=True)
                    crop_anno, _ = resize_normalize(crop_anno, resize_width, interpolate=False)
                    if ratio < 1: print('res ratio', ratio)
                cv2.imwrite(os.path.join(img_crop_dir,base_name+ '_{}.jpg'.format(str(idx))), crop_img)
                cv2.imwrite(os.path.join(anno_crop_dir,base_name+ '_{}.png'.format(str(idx))), crop_anno)
                count+=1
                print('num bbox', count)
        else:
            cv2.imwrite(os.path.join(img_crop_dir, base_name + '_.jpg'), img)
            anno = cv2.cvtColor(anno, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(anno_crop_dir, base_name + '_.png'), anno)
    print('total {} bboxes', count)

if __name__ =='__main__':
    img_dir = '/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/images_jpg'
    anno_dir = '/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/labels_normal'
    crop_source(img_dir, anno_dir)