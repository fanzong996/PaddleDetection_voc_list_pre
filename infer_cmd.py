import os

import cv2
import numpy as np

from utils import logger, config
from utils.predictor import Predictor
from utils.get_image_list import get_image_list
from python.preprocess import create_operators
from python.postprocess import build_postprocess


class ClsPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"])

        self.preprocess_ops = []
        self.postprocess = None
        if "PreProcess" in config:
            if "transform_ops" in config["PreProcess"]:
                self.preprocess_ops = create_operators(config["PreProcess"][
                    "transform_ops"])
        if "PostProcess" in config:
            self.postprocess = build_postprocess(config["PostProcess"])

        # for whole_chain project to test each repo of paddle
        self.benchmark = config["Global"].get("benchmark", False)
        if self.benchmark:
            import auto_logger
            import os
            pid = os.getpid()
            size = config["PreProcess"]["transform_ops"][1]["CropImage"][
                "size"]
            if config["Global"].get("use_int8", False):
                precision = "int8"
            elif config["Global"].get("use_fp16", False):
                precision = "fp16"
            else:
                precision = "fp32"
            self.auto_logger = auto_logger.AutoLogger(
                model_name=config["Global"].get("model_name", "cls"),
                model_precision=precision,
                batch_size=config["Global"].get("batch_size", 1),
                data_shape=[3, size, size],
                save_path=config["Global"].get("save_log_path",
                                               "./auto_log.log"),
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=2)

    def predict(self, images):
        use_onnx = self.args.get("use_onnx", False)
        if not use_onnx:
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
        else:
            input_names = self.predictor.get_inputs()[0].name
            output_names = self.predictor.get_outputs()[0].name

        if self.benchmark:
            self.auto_logger.times.start()
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        if self.benchmark:
            self.auto_logger.times.stamp()

        if not use_onnx:
            input_tensor.copy_from_cpu(image)
            self.predictor.run()
            batch_output = output_tensor.copy_to_cpu()
        else:
            batch_output = self.predictor.run(
                output_names=[output_names],
                input_feed={input_names: image})[0]

        if self.benchmark:
            self.auto_logger.times.stamp()
        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        if self.benchmark:
            self.auto_logger.times.end(stamp=True)
        return batch_output


def main(config):
    cls_predictor = ClsPredictor(config)

    clas_ids_list = []
    scores_str_list = []

    root_list = config["Global"]["infer_imgs"]
    img_list = []
    i = 0
    for root, dirs, files in os.walk(root_list, topdown=True, onerror=None, followlinks=False):
        if i == 0:
            label_list = dirs
            img_list = files
            i = 1
        else:
            img_list.append(files)

    for i in range(24):
        for j in range(len(img_list)):
            infer_list = root_list + '/' + label_list[i] + '/' + img_list[i][j]
            image_list = get_image_list(infer_list)#不能传入list


            batch_imgs = []
            batch_names = []
            cnt = 0
            for idx, img_path in enumerate(image_list):
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(
                        "Image file failed to read and has been skipped. The path: {}".
                        format(img_path))
                else:
                    img = img[:, :, ::-1]
                    batch_imgs.append(img)
                    img_name = os.path.basename(img_path)
                    batch_names.append(img_name)
                    cnt += 1

                if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                                 ) == len(image_list):
                    if len(batch_imgs) == 0:
                        continue
                    batch_results = cls_predictor.predict(batch_imgs)
                    for number, result_dict in enumerate(batch_results):
                        if "PersonAttribute" in config[
                                "PostProcess"] or "VehicleAttribute" in config[
                                    "PostProcess"]:
                            filename = batch_names[number]
                            print("{}:\t {}".format(filename, result_dict))
                        else:
                            clas_ids = result_dict["class_ids"]
                            scores_str = "[{}]".format(", ".join("{:.2f}".format(
                                r) for r in result_dict["scores"]))
                            clas_ids_list.append(clas_ids)
                            scores_str_list.append(scores_str)
                    batch_imgs = []
                    batch_names = []
    return clas_ids_list, scores_str_list

if __name__ == "__main__":
    yaml_dir = "inference_configs/inference_cls.yaml"
    args = config.parse_args(yaml_dir)
    config = config.get_config(args.config, overrides=args.override, show=True)
    clas_ids_list, scores_str_list = main(config)
    print(clas_ids_list[:][1])
    print(scores_str_list[:][1])
