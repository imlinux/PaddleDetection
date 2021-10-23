import cv2

from infer import PredictConfig, Detector


class Predict:

    def __init__(self, model_dir):
        pred_config = PredictConfig(model_dir)

        self.detector = Detector(
            pred_config,
            model_dir,
            use_gpu=False,
            batch_size=1,
            use_dynamic_shape=False,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            cpu_threads=3,
            enable_mkldnn=False)

    def __call__(self, *args, **kwargs):
        return self.detector.predict(*args)


def main():

    model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no"
    img_path = "/home/dong/tmp/dataset/voc/zuowen/JUYE_F_00007.pdf-17.jpg-790-4.jpg"
    img = [cv2.imread(img_path)]

    predict = Predict(model_dir)
    result = predict(img)
    boxes = result["boxes"]
    print(boxes[boxes[:, 1].argsort()[::-1]])


if __name__ == "__main__":
    main()