import argparse
import os
import time

import torch

import pytorch_mask_rcnn as pmr

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val", train=True) # VOC 2012. set train=True for eval
    #d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # COCO 2017
    print(len(d_test))

    print(args)
    num_classes = max(d_test.classes) + 1
    model = pmr.maskrcnn_resnet50(False, num_classes).to(device)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    # print(type(checkpoint))
    # print(checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    #print(checkpoint["eval_info"])
    del checkpoint
    if cuda: torch.cuda.empty_cache()

    print("\nevaluating...\n")
    
    B = time.time()
    eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
    B = time.time() - B
    
    # print(eval_output.get_AP())
    if iter_eval is not None:
        print("\nTotal time of this evaluation: {:.1f} s, speed: {:.1f} imgs/s".format(B, 1 / iter_eval))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc")
    parser.add_argument("--data-dir", default="VOCdevkit/VOC2012")
    parser.add_argument("--ckpt-path", default="results/result_732_scratch_total_ratio_norm_voc_0817/maskrcnn_coco-40.pth")
    # parser.add_argument("--ckpt-path", default="results/result_0627/maskrcnn_voc-20.pth")
    parser.add_argument("--iters", type=int, default=-1) # number of iterations, minus means the entire dataset
    args = parser.parse_args([]) # [] is needed if you're using Jupyter Notebook.
    
    args.use_cuda = True
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)