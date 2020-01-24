import argparse
import os
import pickle
import numpy as np
import time
import multiprocessing as mp
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
from process_mask import ProcessMasks

root_dir = '/data2/datasets_origin/'
fft_save_dir = '/data2/datasets_origin/siw/train_fft/live'
scan_dir = '/data2/datasets_origin/siw/Train/live'

SEQ_LEN = 5
SEQ_STRIP = 3

def key_cmp(v1):
    d1 = int(v1.split("_")[-1].split(".")[-2])
    return d1


class RunPOSFromVideo():
    def __init__(self, sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot

    def __call__(self, source, save_key):
        time1 = time.time()

        mask_process_pipe, chil_process_pipe = mp.Pipe()
        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=False)
            self.plot_process.start()

        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size, save_key=save_key)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source,),
                                    daemon=False)
        mask_processer.start()

        capture = CaptureFrames(self.batch_size, source, show_mask=False)
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()
        time2 = time.time()
        print(f'time {time2 - time1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rppg signal')
    parser.add_argument('--show_flg', default='true', type=bool, help='whether show the visualization result')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    args = parser.parse_args()

    runPOSVid = RunPOSFromVideo(240, 25, 30, False)


    siw_datasets = []

    save_root_key = 'siw/train_fft/live'

    for root, dirs, files in os.walk(scan_dir, topdown=False):
        for name in files:
            if name.split('.')[-1] == 'mov':
                mov_path = os.path.join(root, name)
                class_sess_name = name.split('.')[-2]
                save_key = os.path.join(save_root_key, class_sess_name)
                task_processer = mp.Process(target=runPOSVid, args=(mov_path, save_key),
                                            daemon=False)
                task_processer.start()
                task_processer.join()


    # for k in siw_datasets:
    #     sorted_dataset = sorted(siw_datasets[k], key = key_cmp)
    #     siw_datasets[k] = sorted_dataset

    # runPOSImg = RunPOSFromImg(240, 25, 1, True)
    #
    # for k in siw_datasets:
    #     tp, pe = k.split('_')
    #     if tp != 'real':
    #         continue
    #     save_key = 'siw_v1/train_fft/'
    #     save_key = save_key+'/'+tp+'/'
    #     runPOSImg(siw_datasets[k], save_key)
