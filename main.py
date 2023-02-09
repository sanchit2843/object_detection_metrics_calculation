import os
from src.coco_evaluator import get_coco_summary, get_coco_metrics
from src.bounding_box import BoundingBox
import argparse
from src.enumerators import CoordinatesType, BBType, BBFormat
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import cv2


def get_coco_metrics_from_path(ground_truth_path, detection_path, image_path):
    all_gt_boxes = []
    all_detection_boxes = []
    each_image_metrics = []
    for i in tqdm(os.listdir(ground_truth_path)):
        gt_txt_file = open(os.path.join(ground_truth_path, i), "r")
        if os.path.exists(os.path.join(image_path, i.replace("txt", "png"))):
            h, w, _ = cv2.imread(
                os.path.join(image_path, i.replace("txt", "png"))
            ).shape
        else:
            h, w, _ = cv2.imread(
                os.path.join(image_path, i.replace("txt", "PNG"))
            ).shape
        gt_boxes = []
        detected_boxes = []

        for current_line in gt_txt_file.readlines():
            current_line = current_line.split(" ")
            gt_boxes.append(
                BoundingBox(
                    image_name=i,
                    class_id=current_line[0],
                    coordinates=(
                        float(current_line[1]),
                        float(current_line[2]),
                        float(current_line[3]),
                        float(current_line[4]),
                    ),
                    bb_type=BBType.GROUND_TRUTH,
                    confidence=None,
                    format=BBFormat.YOLO,
                    type_coordinates=CoordinatesType.RELATIVE,
                    img_size=(w, h),
                )
            )

        if os.path.exists(os.path.join(detection_path, i)):
            detection_txt_file = open(os.path.join(detection_path, i), "r")
            for current_line in detection_txt_file.readlines():
                current_line = current_line.split(" ")
                detected_boxes.append(
                    BoundingBox(
                        image_name=i,
                        class_id=current_line[0],
                        coordinates=(
                            float(current_line[1]),
                            float(current_line[2]),
                            float(current_line[3]),
                            float(current_line[4]),
                        ),
                        bb_type=BBType.DETECTED,
                        confidence=float(current_line[5]),
                        format=BBFormat.YOLO,
                        img_size=(w, h),
                        type_coordinates=CoordinatesType.RELATIVE,
                    )
                )
        all_gt_boxes += gt_boxes
        all_detection_boxes += detected_boxes

        # image_metrics = get_coco_summary(gt_boxes, detected_boxes)
        # image_metrics_list = [i]
        # for _, v in image_metrics.items():
        #     if math.isnan(v):
        #         image_metrics_list.append(-1)
        #         continue
        #     image_metrics_list.append(v)
        # each_image_metrics.append(np.array(image_metrics_list))

    all_image_metrics = get_coco_summary(all_gt_boxes, all_detection_boxes)
    print(all_image_metrics)
    a = get_coco_metrics(all_gt_boxes, all_detection_boxes)

    return each_image_metrics, all_image_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_gt",
        "-p",
        type=str,
        default="./example_result_folder",
        required=False,
        help="Path to result folder in structure defined in readme",
    )
    parser.add_argument(
        "--path_to_pred",
        type=str,
        default="./example_result_folder",
        required=False,
        help="Path to result folder in structure defined in readme",
    )

    parser.add_argument(
        "--path_to_png",
        type=str,
        default="./example_result_folder",
        required=False,
        help="Path to result folder in structure defined in readme",
    )

    args = parser.parse_args()

    each_image_metrics, all_image_metrics = get_coco_metrics_from_path(
        args.path_to_gt, args.path_to_pred, args.path_to_png
    )
    print(all_image_metrics)
    # image_metrics_list = ["all_images"]

    # for _, v in all_image_metrics.items():
    #     if math.isnan(v):
    #         image_metrics_list.append(-1)
    #         continue
    #     image_metrics_list.append(v)

    # each_image_metrics.append(np.array(image_metrics_list))
    # each_image_metrics = pd.DataFrame(np.array(each_image_metrics))
    # each_image_metrics.columns = ["image name"] + list(all_image_metrics.keys())

    # each_image_metrics.to_csv(
    #     os.path.join(args.path_to_results, "each_image_results.csv"), index=False
    # )

    # print("object detection coco metrics for all images")
    # print(all_image_metrics)
