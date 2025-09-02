import cv2
import os
import re

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <filename>{}</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''


labels = ['no-vest', 'ok', 'vest']  

cnt = 0
txt_path = r'E:\DATASET\111\voc2yolo\train2017'  
image_path = r'E:\DATASET\COCO_hat\train2017'     
path = r'E:\DATASET\111\yolo2voc\train2017'       

for root, dirs, files in os.walk(image_path):  
    for ft in files:
        filename, ext = os.path.splitext(ft)
        ftxt = filename + '.txt'
        fxml = filename + '.xml'
        img_path = os.path.join(root, ft)
        txt_file_path = os.path.join(txt_path, ftxt)
        xml_file_path = os.path.join(path, fxml)
        obj = ''

      
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖像 {img_path}，跳過。")
            continue
        img_h, img_w = img.shape[:2]
        depth = img.shape[2] if len(img.shape) > 2 else 1
        head = xml_head.format(ft, img_w, img_h, depth)

     
        if not os.path.exists(txt_file_path):
            print(f"標記文件 {txt_file_path} 不存在，跳過。")
            continue

    
        with open(txt_file_path, 'r') as f:
            for line in f.readlines():
                
                yolo_data = re.findall(r"[\d\.]+", line)
                if len(yolo_data) != 5:
                    print(f"{txt_file_path} 格式不正確，跳過此行：{line.strip()}")
                    continue
                label_idx = int(float(yolo_data[0]))
                if label_idx >= len(labels):
                    print(f"{txt_file_path} 中的標籤索引 {label_idx} 超出範圍，跳過此行。")
                    continue
                label = labels[label_idx]
                center_x = float(yolo_data[1]) * img_w
                center_y = float(yolo_data[2]) * img_h
                bbox_width = float(yolo_data[3]) * img_w
                bbox_height = float(yolo_data[4]) * img_h

                xmin = int(center_x - bbox_width / 2)
                ymin = int(center_y - bbox_height / 2)
                xmax = int(center_x + bbox_width / 2)
                ymax = int(center_y + bbox_height / 2)

                # 确保坐标在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)

                obj += xml_obj.format(label, xmin, ymin, xmax, ymax)

        # 写入XML文件
        with open(xml_file_path, 'w') as f_xml:
            f_xml.write(head + obj + xml_end)
        cnt += 1
        print(f"已處理 {cnt}：{xml_file_path}")
