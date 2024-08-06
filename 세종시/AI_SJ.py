from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import glob
import os
import argparse

# detect
model_detect = YOLO('./detect_best.pt')

# segment
model_seg = YOLO('./segment_best.pt') 

# 분석 결과 표기 방법
# [라벨] [객체 중심좌표X] [ 객체 중심좌표Y] [객체 너비(W)] [객체 높이(H)]
# 라벨 0 = 무단횡단 보행자 / 1 = 횡단 보행자
# 일반 보행자는 출력 x

def intersection(seg_coords_list, point):
    point = Point(point)
    circle = point.buffer(0.05) # 보정값
    
    for seg_coords in seg_coords_list:
        polygon = Polygon(seg_coords)
        intersection_area = polygon.intersection(circle)
        if not intersection_area.is_empty:
            return True  # 교집합이 있는 경우
    
    return False  # 교집합이 없는 경우

def create_result_dir(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print("Result directory created successfully.")
        
def result_img(data_dir, result_dir):
    if not os.path.exists(data_dir):
        print("Error: The specified data directory does not exist.")
        return
    
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))

    print("Image files in the directory:")
    print(image_files)
    
    
    for image_file in image_files:    
        detect_result = model_detect(image_file, conf=0.3, agnostic_nms=True, augment=True, iou=0.5, visualize=True, retina_masks=True, show=True)[0]
        detect_boxes = detect_result.boxes # detect
        
        seg_result = model_seg(image_file, agnostic_nms=True, retina_masks=True, iou=0.7)[0]
        seg_boxes = seg_result.boxes # segment
        seg_masks = seg_result.masks

        road_coords = []
        road_coords_list = []
        crosswalk_coords = []
        crosswalk_coords_list = []
                
        # segment
        for mask, box in zip(seg_masks, seg_boxes):
            if int(box.cls.item()) == 2: # 도로: # 도로
                road_coords = mask.xyn[0]
                road_coords_list.append(road_coords)
                
            if int(box.cls.item()) == 1: # 도로: # 횡단보도
                crosswalk_coords = mask.xyn[0]
                crosswalk_coords_list.append(crosswalk_coords)
        
        result_bboxes = []
        
        # detect
        for box in detect_boxes:
            detect_confidence = box.conf[0]
            
            if detect_confidence:
                xn, yn, wn, hn = box.xywhn[0].float().tolist() # 정규화 좌표
                point_coords = [((box.xyxyn[0][0].item() + box.xyxyn[0][2].item()) / 2), box.xyxyn[0][3].item()] # xyxyn

                # 교집합 확인
                on_road = intersection(road_coords_list, point_coords)
                on_crosswalk = intersection(crosswalk_coords_list, point_coords)
                cls = None
                
                if on_crosswalk is True:
                    cls = 1
                
                elif on_road is True:
                    cls = 0
                    
                elif on_crosswalk & on_road is False: 
                    cls = None

                # 텍스트 파일 출력    
                file_name = os.path.splitext(os.path.basename(image_file))[0]
                result_bboxes.append([cls, xn, yn, wn, hn])
                result_txt = open(os.path.join(result_dir, file_name + '.txt'), 'w')                
                
                if len(result_bboxes) == 0:
                    result_txt.close()
                    
                else:
                    for bbox in result_bboxes:
                        if bbox[0] is not None:
                            result_txt.write("{:d} {:f} {:f} {:f} {:f}\n".format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))
                    result_txt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments to list image files and create result directory.")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing image files.", required=True)
    parser.add_argument("--result_dir", type=str, help="Path to the directory to store analysis result text files.", required=True)
    args = parser.parse_args()
    
    create_result_dir(args.result_dir)
    result_img(args.data_dir, args.result_dir)