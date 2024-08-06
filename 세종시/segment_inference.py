from ultralytics import YOLO
from PIL import Image
import numpy as np

def inference_model():

    #segment_best_model
    #model = YOLO("C:/Users/amuse/PycharmProjects/pythonProject3/runs/segment/train12/weights/best.pt")  # load a pretrained model (recommended for training)
    #model.add_callback("on_train_start", freeze_layer)

    #detect_model
    model = YOLO("/home/smu_01/runs/segment/train3/weights/best.pt")  # best_model
    #results = model.predict(source='D:/Crosswalk.v3i.yolov8/test/images/', conf=0.25)
    results = model.predict("/home/smu_01/test/images/", conf = 0.25)

    
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

        im.save("/home/smu_01/result/" + str(r) + 'results.jpg')  # save image
    
if __name__ == '__main__':
    inference_model()