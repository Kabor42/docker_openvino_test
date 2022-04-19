"""hogy egy, csak a képbeolvasást és detekciós algoritmusokat futtató, *szinkron* API-t használó példaprogramot dobjatok össze és futassátok."""

import logging
import os
import sys
import time

import cv2 as cv
import numpy as np
from openvino.inference_engine import ie_api
from dataclasses import dataclass, astuple
from typing import Dict, List, Any, Tuple

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)


@dataclass
class BoundingBox:
    # TODO hosszan kiirni
    tl_x: int
    tl_y: int
    br_x: int
    br_y: int

    def get_width(self) -> int:
        return self.br_x - self.tl_x

    def get_height(self) -> int:
        return self.br_y - self.br_x

    def get_center(self) -> Tuple[int, int]:
        x = self.tl_x + self.get_width() // 2
        y = self.tl_y + self.get_height() // 2
        return x, y

    def get_area(self) -> int:
        return self.get_width() * self.get_height()

    def __array__(self):
        return np.array(astuple(self))

    def __len__(self):
        return astuple(self).__len__()

    def __getitem__(self, item):
        return astuple(self).__getitem__(item)

    def __str__(self):
        return f"[{self.tl_x},{self.tl_y},{self.br_x},{self.br_y}]"


@dataclass
class ObjectDetectionResult(BoundingBox):
    confidence: float
    id: int
    label: int

    def __str__(self):
        return super().__str__() + f" {self.confidence:.2%}"


@dataclass
class NetworkInfo:
    name: str
    layout: str
    shape: List[int]
    precision: str


def get_inputs_info(exec_net: ie_api.ExecutableNetwork) -> List[NetworkInfo]:
    input_infos = list(exec_net.input_info.keys())
    input_datas = [exec_net.input_info[ii].input_data for ii in input_infos]
    inputs_list = []
    for input_data in input_datas:
        inputs_list.append(
            NetworkInfo(
                name=input_data.name,
                shape=input_data.shape,
                precision=input_data.precision,
                layout=input_data.layout,
            )
        )
    return inputs_list


def get_outputs_info(exec_net: ie_api.ExecutableNetwork) -> List[NetworkInfo]:
    output_infos = list(exec_net.outputs.keys())
    output_list = []
    for output_info in output_infos:
        output_data = exec_net.outputs[output_info]
        output_list.append(
            NetworkInfo(
                name=output_data.name,
                shape=output_data.shape,
                precision=output_data.precision,
                layout=output_data.layout,
            )
        )
    return output_list


def preprocess(frame, input_info: NetworkInfo) -> Dict[str, Any]:
    img = cv.resize(frame, (input_info.shape[-1], input_info.shape[-2]))
    if input_info.layout[-2:] == "HW":
        img = img.transpose((2, 0, 1))
    else:
        logging.debug(f"Layout: {input_info.layout}")
        raise NotImplementedError("Unknown network layout.")
    if input_info.layout[:2] == "NC":
        img = np.expand_dims(img, axis=0)
    return {input_info.name: img}


def postprocess(
    result: Dict[str, Any],
    original_dimensions: Tuple[int, int],
    output_info: NetworkInfo,
) -> Any:
    processed_result = result[output_info.name].reshape(-1, output_info.shape[-1])
    filtered_result = np.array([d for d in processed_result if d[2] >= 0.7])
    retval = []
    for r in filtered_result:
        retval.append(
            ObjectDetectionResult(
                id=int(r[0]),
                label=int(r[1]),
                confidence=float(r[2]),
                tl_x=max(0, int(r[3] * original_dimensions[1])),
                tl_y=max(0, int(r[4] * original_dimensions[0])),
                br_x=min(original_dimensions[1], int(r[5] * original_dimensions[1])),
                br_y=min(original_dimensions[0], int(r[6] * original_dimensions[0])),
            )
        )
    return retval


def main():
    model = "person-detection-0202"
    model_path = os.path.sep.join([os.environ["HOME"], "models", model, model])

    logging.info("Loading camera.")
    cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
    if not cap.isOpened():
        logging.error("Cannot open camera.")
        sys.exit()
    logging.info("Camera succesfully loaded.")
    logging.info("Loading OpenVINO Core")
    core = ie_api.IECore()
    logging.info(f"Reading model {model}")
    net = core.read_network(model=model_path + ".xml", weights=model_path + ".bin")
    logging.info("Loading model to device.")
    exec_net = core.load_network(network=net, device_name="MYRIAD")
    logging.info("OpenVINO Setup done.")

    input_info = list(
        filter(lambda i: i.name in ["data", "image"], get_inputs_info(exec_net))
    )[0]
    output_info = list(
        filter(
            lambda i: i.name in ["detection_out", "boxes"], get_outputs_info(exec_net)
        )
    )[0]
    logging.info(f"{model} input info {input_info}")
    logging.info(f"{model} output info {output_info}")

    logging.info("Starting loop")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading frame from source.")
                continue
            original_dimensions = frame.shape[:2]

            payload = preprocess(frame, input_info)
            raw_result = exec_net.infer(inputs=payload)
            detections = postprocess(raw_result, original_dimensions, output_info)
            # print(len(detections),end='\r')
            time.sleep(0.05)
    except cv.error as e:
        logging.exception(f"Camera exception: {e}")
    cap.release()
    logging.info("Exiting loop")


if __name__ == "__main__":
    main()
