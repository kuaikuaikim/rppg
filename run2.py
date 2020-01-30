import cv2
import numpy as np
from mask_frame import MaskFrame
from process_rppg import ProcessRppg

from utils import *
import sys
from optparse import OptionParser




def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default=0,
                      help='Signal Source: 0 for webcam or file path')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=30,
                      type='int', help='batch size')
    parser.add_option('-f', '--frame-rate', dest='framerate', default=25,
                      type='int', help='Frame Rate')

    (options, _) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    source = args.source
    # runPOS = RunPOS(270, args.framerate, args.batchsize, True)
    camera = cv2.VideoCapture(source)

    (grabbed, frame) = camera.read()

    mask_frame = MaskFrame(args.batchsize)
    process_rppg = ProcessRppg()

    while grabbed:
        (grabbed, orig) = camera.read()
        if not grabbed:
            break
        data = mask_frame.get_mask_frame(orig)
        process_rppg(data)

