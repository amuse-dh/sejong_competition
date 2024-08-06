from ultralytics import YOLO
from PIL import Image
import numpy as np

def inference_model():

    #segment_best_model
    #model = YOLO("C:/Users/amuse/PycharmProjects/pythonProject3/runs/segment/train12/weights/best.pt")  # load a pretrained model (recommended for training)
    #model.add_callback("on_train_start", freeze_layer)

    #detect_model
    model = YOLO("C:/Users/amuse/Downloads/best.pt")  # best_model
    #results = model.predict(source='D:/Crosswalk.v3i.yolov8/test/images/', conf=0.25)
    results = model.predict('D:/human_detect.v3i.yolov8/test/images/', conf = 0.25, iou = 0.5, augment = True, agnostic_nms = True, retina_masks = True)


    for r in range(len(results)):

        im_array = results[r].plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        '''
        for idx in range(len(results[r])):
            mask = np.asarray(results[r][idx].masks)
            print(mask.shape)

            img = mask * im
            img.save("D:/Crosswalk.v3i.yolov8/test/result/"+str(r)+str("_")+str(idx)+'results.jpg')  # save image
        '''

        im.save("D:/human_detect.v3i.yolov8/test/detect/" + str(r) + 'results.jpg')  # save image

if __name__ == '__main__':
    inference_model()