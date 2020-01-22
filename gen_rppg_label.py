import argparse
import os
import pickle
import numpy as np

root_dir = '/data2/datasets/'
scan_dir = '/data2/datasets/siw_m/train_image'

SEQ_LEN = 5
SEQ_STRIP = 3

def my_cmp(v1, v2):
    d1 = int(v1.split("_")[-2].split(".")[-2])
    d2 = v2(v1.split("_")[-2].split(".")[-2])
    return d1 < d2

def key_cmp(v1):
    d1 = int(v1.split("_")[-1].split(".")[-2])
    return d1


class POSLabel():
    def __init__(self, sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot

    def __call__(self, source):
        time1 = time.time()

        mask_process_pipe, chil_process_pipe = mp.Pipe()
        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=False)
            self.plot_process.start()

        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source,),
                                    daemon=False)
        mask_processer.start()

        capture = CaptureFrames(self.batch_size, source, show_mask=True)
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()
        time2 = time.time()
        time2 = time.time()
        print(f'time {time2 - time1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rppg signal')
    parser.add_argument('--show_flg', default='true', type=bool, help='whether show the visualization result')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    args = parser.parse_args()

    siw_datasets = {}
    for root, dirs, files in os.walk(scan_dir, topdown=False):
        for name in files:
            img_class = root.split("/")[-2]
            people_id = root.split("/")[-1]
            class_key = img_class + '_' + people_id
            if class_key not in siw_datasets:
                siw_datasets[class_key] = []
            cur_file = os.path.join(root, name)
            cur_file = cur_file.replace("/data2/datasets/","")
            siw_datasets[class_key].append(cur_file)


    for k in siw_datasets:
        sorted_dataset = sorted(siw_datasets[k], key = key_cmp)
        siw_datasets[k] = sorted_dataset

    seq_id = 0
    seq_datasets = []
    for k in siw_datasets:
        step = 0
        seq_class = k.split("_")[-2]
        if seq_class == 'real':
            label = 1
        else:
            label = 0

        while (step+SEQ_LEN) < len(siw_datasets[k]):
            seq = {}
            seq['seq'] = []
            seq['label'] = label
            dirty_data = False
            for inc in range(SEQ_LEN):
                if label == 1:
                    cur_line = siw_datasets[k][step+inc]
                    cur_depth_path = cur_line.replace("train_image", "train_depth_32")
                    cur_depth_path = cur_depth_path.replace("png", "npy")
                    cur_depth_path = os.path.join(root_dir, cur_depth_path)
                    depth_data = np.load(cur_depth_path)
                    if np.isnan(depth_data).any():
                        dirty_data = True
                        break
                    seq['seq'].append(siw_datasets[k][step + inc])
                else:
                    seq['seq'].append(siw_datasets[k][step+inc])
            if not dirty_data:
                seq_datasets.append(seq)
            step += SEQ_STRIP

    f = open("./siw_seq_b5s3.pick", 'wb')
    pickle.dump(seq_datasets, f)
    f.close()