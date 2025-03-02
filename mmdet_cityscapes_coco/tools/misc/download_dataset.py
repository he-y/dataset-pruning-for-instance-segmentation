import argparse
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile
import subprocess

import torch
from mmengine.utils.path import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download datasets for training')
    parser.add_argument(
        '--dataset-name', type=str, help='dataset name', default='coco2017')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='data/coco')
    parser.add_argument(
        '--delete',
        action='store_true',
        help='delete the download zipped files')
    parser.add_argument(
        '--threads', type=int, help='number of threading', default=4)
    parser.add_argument(
        '--username',
        type=str,
        help='Cityscapes username (required for Cityscapes)',
        default=None)
    parser.add_argument(
        '--password',
        type=str,
        help='Cityscapes password (required for Cityscapes)',
        default=None)
    args = parser.parse_args()
    return args


def download(url, dir, unzip=True, delete=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def download_cityscapes(dir, username, password):
    """Handle Cityscapes dataset download, which requires login."""
    if username is None or password is None:
        print("Cityscapes requires a username and password. Please provide them.")
        return

    # Define the login and download commands
    login_cmd = [
        "wget", "--keep-session-cookies", "--save-cookies=cookies.txt",
        "--post-data", f"username={username}&password={password}&submit=Login",
        "https://www.cityscapes-dataset.com/login/"
    ]
    files_to_download = [
        "https://www.cityscapes-dataset.com/file-handling/?packageID=2",
        "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
    ]
    download_cmds = [
        [
            "wget", "--load-cookies", "cookies.txt", "--content-disposition",
            url
        ] for url in files_to_download
    ]

    # Ensure the save directory exists
    dir = Path(dir)
    mkdir_or_exist(dir)
    subprocess.run(login_cmd, check=True)

    # Download each file
    for cmd in download_cmds:
        print(f"Downloading {cmd[-1]} to {dir}")
        subprocess.run(cmd, cwd=dir, check=True)

    # Cleanup
    Path("cookies.txt").unlink(missing_ok=True)
    print("Cityscapes download complete.")

def main():
    args = parse_args()
    path = Path(args.save_dir)

    if args.dataset_name == 'coco2017':
        path =  Path('data/coco')
    elif args.dataset_name == 'cityscapes':
        path =  Path('data/cityscapes')
    elif args.dataset_name == 'voc2012':
        path = Path('../voc_2012')
        

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        # TODO: Support for downloading Panoptic Segmentation of COCO
        coco2017=[
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip',
            'http://images.cocodataset.org/zips/test2017.zip',
            'http://images.cocodataset.org/zips/unlabeled2017.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip',  # noqa
        ],
        coco2014=[
            'http://images.cocodataset.org/zips/train2014.zip',
            'http://images.cocodataset.org/zips/val2014.zip',
            'http://images.cocodataset.org/zips/test2014.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2014.zip'  # noqa
        ],
        lvis=[
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
        ],
        voc2007=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',  # noqa
        ],
        voc2012=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # noqa
        ],
        balloon=[
            # src link: https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip # noqa
            'https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip'
        ],
        # Note: There is no download link for Objects365-V1 right now. If you
        # would like to download Objects365-V1, please visit
        # http://www.objects365.org/ to concat the author.
        objects365v2=[
            # training annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz',  # noqa
            # validation annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json',  # noqa
            # training url root
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/',  # noqa
            # validation url root_1
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/',  # noqa
            # validation url root_2
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/'  # noqa
        ],
        ade20k_2016=[
            # training images and semantic segmentation annotations
            'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',  # noqa
            # instance segmentation annotations
            'http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar',  # noqa
            # img categories ids
            'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/imgCatIds.json',  # noqa
            # category mapping
            'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/categoryMapping.txt'  # noqa
        ],
        refcoco=[
            # images
            'http://images.cocodataset.org/zips/train2014.zip',
            # refcoco annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',
            # refcoco+ annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip',
            # refcocog annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip'
        ],
        cityscapes=None
)
    url = data2url.get(args.dataset_name, None)

    if args.dataset_name == "cityscapes":
        download_cityscapes(path, args.username, args.password)
        return

    if url is None:
        print('Only support ADE20K, COCO, RefCOCO, VOC, LVIS, '
              'balloon, and Cityscapes now.')
        return

    download(
        url,
        dir=path,
        delete=args.delete,
        threads=args.threads)


if __name__ == '__main__':
    main()