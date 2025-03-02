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
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "train", train=True) # VOC 2012. set train=True for score calculation
    # d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # COCO 2017

    print(args)
    num_classes = max(d_test.classes) + 1

    if 'coco' in args.dataset:
        model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    else:
        model = pmr.maskrcnn_resnet50(False, num_classes).to(device)

    if args.get_score == 'forgetting_pixel':
        import bisect

        args.lr = 0.02 * 1 / 16
        args.momentum = 0.9
        args.weight_decay = 0.0001
        args.iters = 2000
        args.warmup_iters = max(1000, len(d_test))
        args.epochs = 20
        args.lr_steps = [6, 7]

        start_epoch = 0
        d_train = d_test
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)

        for epoch in range(start_epoch, args.epochs):
            model.head.reset_forgetting_tracking(len(d_train)) 
            print("\nepoch: {}".format(epoch + 1))
                
            A = time.time()
            args.lr_epoch = lr_lambda(epoch) * args.lr
            print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
            iter_train = pmr.get_roi_scores_fogetting(d_train, model, device, optimizer, epoch, args)
            A = time.time() - A

    elif 'ratio' in args.get_score:
        start_time = time.time()
        results = pmr.get_p2a_ratio_scores(d_test, model, device, get_score=args.get_score)
        end_time = time.time()  # End timing
        print("# --------------------------- #")
        print(f"Function execution took: {end_time - start_time} seconds")
        print("# --------------------------- #")

    else:
        if 'voc' in args.dataset:
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            del checkpoint

        if cuda: torch.cuda.empty_cache()

        print("\ncalculating...\n")
        
        start_time = time.time()
        # eval_output, iter_eval = pmr.test_trainset_score_cal(model, d_test, device, args)
        results = pmr.get_roi_scores(d_test, model, device, get_score=args.get_score)  # (data_loader, model, device)
        end_time = time.time()  # End timing
        print("# --------------------------- #")
        print(f"Function execution took: {end_time - start_time} seconds")
        print("# --------------------------- #")
        
        # print(eval_output.get_AP())
        # if iter_eval is not None:
        #     print("\nTotal time of this calculation: {:.1f} s, speed: {:.1f} imgs/s".format(B, 1 / iter_eval))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc")
    parser.add_argument("--data-dir", default="VOCdevkit/VOC2012")
    parser.add_argument("--ckpt-path", default="results/result_0627/maskrcnn_voc-20.pth")
    parser.add_argument("--iters", type=int, default=20) 

    # ratio
    # roi, boundary_roi, el2n, boundary_el2n
    # forgetting, aum
    parser.add_argument("--get_score", type=str, default='ratio') 
    # parser.add_argument("--get_score", type=str, default='el2n') 

    args = parser.parse_args([]) # [] is needed if you're using Jupyter Notebook.
    
    args.use_cuda = True
    # args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.txt")
    
    main(args)