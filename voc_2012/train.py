import bisect
import glob
import os
import re
import time

import torch

import pytorch_mask_rcnn as pmr 
import matplotlib.pyplot as plt
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train", train=True)

    if args.voc_select_num > 0:
        if args.random:
            dataset_train.sample_images(num_samples=args.voc_select_num)
            print('Random sample ...')
        else:
            # dataset_train.sample_top_k_images(k=args.voc_select_num)
            # ratio
            if args.rank_method == 'ratio':
                print('Ranking by ratio ...')
                dataset_train.sort_images_by_class_ratio(top_k=args.voc_select_num)
            elif args.rank_method == 'total_ratio':
                print('Ranking by total ratio ...')
                dataset_train.sort_images_by_total_ratio(top_k=args.voc_select_num)
            elif args.rank_method == 'total_ratio_norm':
                print('Ranking by total ratio norm ...')
                dataset_train.sort_images_by_total_ratio_norm(top_k=args.voc_select_num)
            elif args.rank_method == 'total_norm_ratio':
                print('Ranking by total norm ratio ...')
                dataset_train.sort_images_by_total_norm_ratio(top_k=args.voc_select_num)
            elif args.rank_method == 'total_norm_ratio_norm':
                print('Ranking by total norm ratio norm ...')
                dataset_train.sort_images_by_total_norm_ratio_norm(top_k=args.voc_select_num)
            elif args.rank_method == 'total_norm_ratio_norm_count':
                print('Ranking by total norm ratio norm by count ...')
                dataset_train.sort_images_by_total_norm_ratio_count(top_k=args.voc_select_num)
            elif args.rank_method == 'total_boundary_ratio':
                print('Ranking by total boundary ratio ...')
                dataset_train.sort_images_by_total_boundary_ratio(top_k=args.voc_select_num)
            elif args.rank_method == 'total_boundary_ratio_norm':
                print('Ranking by total boundary ratio norm ...')
                dataset_train.sort_images_by_total_boundary_ratio_norm(top_k=args.voc_select_num)
    

            # entropy
            elif args.rank_method == 'roi_score':
                print('Ranking by RoI score ...')
                dataset_train.sort_images_by_class_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_score':
                print('Ranking by RoI total score ...')
                dataset_train.sort_images_by_total_roi_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_boundary_score':
                print('Ranking by RoI total boundary score ...')
                dataset_train.sort_images_by_total_roi_boundary_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_boundary_score_norm':
                print('Ranking by RoI total boundary score norm ...')
                dataset_train.sort_images_by_total_roi_boundary_norm_score(top_k=args.voc_select_num)

            # el2n
            elif args.rank_method == 'roi_total_el2n_score':
                print('Ranking by RoI total el2n score ...')
                dataset_train.sort_images_by_total_roi_el2n_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_boundary_el2n_score':
                print('Ranking by RoI total boundary el2n score ...')
                dataset_train.sort_images_by_total_roi_el2n_boundary_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_el2n_score_norm':
                print('Ranking by RoI total boundary el2n score ...')
                dataset_train.sort_images_by_total_roi_el2n_norm_score(top_k=args.voc_select_num)
            elif args.rank_method == 'roi_total_boundary_el2n_score_norm':
                print('Ranking by RoI total boundary el2n score ...')
                dataset_train.sort_images_by_total_roi_el2n_boundary_norm_score(top_k=args.voc_select_num)

            # aum
            elif args.rank_method == 'aum_total_score':
                print('Ranking by AUM total score ...')
                dataset_train.sort_images_by_total_aum_score(top_k=args.voc_select_num)
            elif args.rank_method == 'aum_total_score_reverse':
                print('Ranking by AUM reverse total score ...')
                dataset_train.sort_images_by_total_aum_score_reverse(top_k=args.voc_select_num)
            elif args.rank_method == 'aum_total_score_norm_reverse':
                print('Ranking by AUM reverse total score norm ...')
                dataset_train.sort_images_by_total_aum_reverse_norm(top_k=args.voc_select_num)

            # forgetting
            elif args.rank_method == 'forgetting_total_score':
                print('Ranking by RoI total score ...')
                dataset_train.sort_images_by_total_forgetting_score(top_k=args.voc_select_num)

            # ccs
            elif args.rank_method == 'ratio_scale':
                print('Ranking by ratio_scale score ...')
                dataset_train.sort_images_by_scale_and_ratio(top_k=args.voc_select_num)
            elif args.rank_method == 'ratio_ccs':
                print('Ranking by ratio score with CCS V1...')
                dataset_train.sort_images_by_class_ratio_ccs(top_k=args.voc_select_num, skip_ratio=0.1)
            elif args.rank_method == 'score_ccs':
                print('Ranking by score score with CCS V1...')
                dataset_train.sort_images_by_class_score_ccs(top_k=args.voc_select_num, skip_ratio=0.15)
            elif args.rank_method == 'ratio_ccs_v2':
                print('Ranking by ratio score with CCS V2...')
                dataset_train.sort_images_by_class_ratio_ccs_v2(top_k=args.voc_select_num, skip_ratio=0.10)

            elif args.rank_method == 'stratified_roi_score':
                print('Ranking by roi score with stratified CCS V3...')
                dataset_train.stratified_sampling_by_roi_score(top_k=args.voc_select_num, pruning_hard_rate=0)
            elif args.rank_method == 'stratified_el2n':
                print('Ranking by EL2N score with stratified CCS V3...')
                dataset_train.stratified_sampling_by_roi_score(top_k=args.voc_select_num, pruning_hard_rate=0)
            elif args.rank_method == 'stratified_aum':
                print('Ranking by AUM score with stratified CCS V3...')
                start_time = time.time()

                dataset_train.stratified_sampling_by_aum_score(top_k=args.voc_select_num, pruning_hard_rate=0.1)
                
                end_time = time.time()  # End timing
                print("# --------------------------- #")
                print(f"Function execution took: {end_time - start_time} seconds")
                print("# --------------------------- #")

            elif args.rank_method == 'stratified_reverse_aum':
                print('Ranking by reverse AUM score with stratified CCS V3...')
                dataset_train.stratified_sampling_by_reverse_aum_score(top_k=args.voc_select_num, pruning_hard_rate=0)

            

    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)

    print('Train set size: ', len(d_train))
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val", train=True) # set train=True for eval
    print('Test set size: ', len(d_test))
        
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = max(d_train.dataset.classes) + 1 # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        # pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        # print(eval_output.get_AP())

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 2
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="./data/coco2017")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")

    parser.add_argument("--dir_path", type=str, default=None, help="saved path")
    parser.add_argument("--random", type=bool, default=False, help="keep number of objects")
    parser.add_argument("--voc_select_num", type=int, default=-1, help="keep number of objects")
    parser.add_argument("--rank_method", type=str, default='roi_score', help="Method rank the object")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        if args.dir_path is not None:
            dir_path = args.dir_path
        else:
            dir_path = "./results/result_voc_maskrcnn_fpn"
        print('Saved at: ', dir_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        args.ckpt_path = os.path.join(dir_path, "maskrcnn_{}.pth".format(args.dataset))
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
    
