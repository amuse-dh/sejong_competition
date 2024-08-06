from ultralytics import YOLO

def valid_model():


    #detect_model
    model = YOLO("C:/Users/amuse/PycharmProjects/pythonProject3/runs/segment/best.pt")  # load a pretrained model (recommended for training)
    #model = YOLO("C:/Users/amuse/PycharmProjects/pythonProject3/runs/detect/train9/weights/best.pt")
    #model.add_callback("on_train_start", freeze_layer)

    #detect
    #result = model.train(data="C:/Users/amuse/Downloads/human.v1i.yolov8/data.yaml", epochs=20, imgsz=640, lr0=0.00001, batch=32, workers=8)
    validation_results = model.val(data='C:/Users/amuse/OneDrive/바탕 화면/Crosswalk.v3i.yolov8 (1)/data.yaml',
                                   imgsz=640,
                                   batch=82,
                                   conf=0.2,
                                   iou=0.6,
                                   device='0')


    #metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    valid_model()


