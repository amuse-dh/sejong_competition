from ultralytics import YOLO

# Load a model
def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")

def train_model():


    #detect_model
    model = YOLO("D:/yolov8x.pt")  # load a pretrained model (recommended for training)
    #model = YOLO("C:/Users/amuse/PycharmProjects/pythonProject3/runs/detect/train9/weights/best.pt")
    model.add_callback("on_train_start", freeze_layer)

    #detect
    #result = model.train(data="C:/Users/amuse/Downloads/human.v1i.yolov8/data.yaml", epochs=20, imgsz=640, lr0=0.00001, batch=32, workers=8)
    result = model.train(data="C:/Users/amuse/Downloads/person.v5i.yolov8/data.yaml", epochs=1000, imgsz=640, batch=64, workers=8)


    #metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    train_model()