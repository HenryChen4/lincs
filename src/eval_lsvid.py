from __future__ import print_function, absolute_import

import glob
import os.path as osp

import numpy as np

import torch
import json
import sys
import os
import pickle as pkl
import fire
import torch.nn.functional as F

from osnet.torchreid.data.datasets.video.mars import Mars # made some adjustments
from osnet.torchreid.data.transforms import RandomErasing, Random2DTranslation, RandomHorizontalFlip, RandomPatch, build_transforms
from osnet.torchreid.utils.feature_extractor import FeatureExtractor

from osnet.torchreid.models.osnet import OSNet, OSBlock, init_local_ckpt, prep_img_paths

from osnet.torchreid.utils.reidtools import visualize_ranked_results

from tqdm import tqdm
from datetime import datetime

from torchreid.data import ImageDataset

EXPERIMENTS = {
    "osnet_market": {
        "result_subdir_ext": "lsvid_market",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    },
    "osnet_duke": {
        "result_subdir_ext": "lsvid_duke",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    },
    "osnet_msmt": {
        "result_subdir_ext": "lsvid_msmt",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    },
    "osnet_combine": {
        "result_subdir_ext": "lsvid_combine",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    }
}

class LSVID(object):
    """
    LS-VID

    Official protocol:
    - train split from `list_sequence/list_seq_train.txt`
    - query split from `test/query.npy`
    - gallery is the full LS-VID test tracklet set
    """

    dataset_dirname = 'LS-VID'

    def __init__(self, root, min_seq_len=0, verbose=True, **kwargs):
        normalized_root = osp.normpath(root)
        if osp.basename(normalized_root) == self.dataset_dirname:
            self.root = normalized_root
        else:
            self.root = osp.join(normalized_root, self.dataset_dirname)

        self.list_seq_train_path = osp.join(self.root, 'list_sequence', 'list_seq_train.txt')
        self.list_seq_test_path = osp.join(self.root, 'list_sequence', 'list_seq_test.txt')
        self.query_idx_path = osp.join(self.root, 'test', 'query.npy')
        self.info_test_path = osp.join(self.root, 'test', 'info_test.npy')

        self._check_before_run()

        test_camids = self._load_test_camids()

        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_sequence_list(
            self.list_seq_train_path,
            relabel=True,
            min_seq_len=min_seq_len,
        )
        test_tracklets, num_test_tracklets, num_test_pids, num_test_imgs = self._process_sequence_list(
            self.list_seq_test_path,
            relabel=False,
            min_seq_len=min_seq_len,
            camid_lookup=test_camids,
        )

        query_idx = np.load(self.query_idx_path, allow_pickle=True).astype(np.int64).reshape(-1) - 1
        if len(query_idx) == 0:
            raise RuntimeError("No query indices found in '{}'".format(self.query_idx_path))
        if query_idx.min() < 0 or query_idx.max() >= len(test_tracklets):
            raise RuntimeError("Query indices in '{}' are out of range".format(self.query_idx_path))

        query = [test_tracklets[idx] for idx in query_idx.tolist()]
        gallery = test_tracklets

        num_query_tracklets = len(query)
        num_query_pids = len({pid for _, pid, _ in query})
        num_gallery_tracklets = len(gallery)
        num_gallery_pids = num_test_pids

        num_imgs_per_tracklet = num_train_imgs + num_test_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_test_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets

        if verbose:
            print("=> LS-VID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  protocol: query subset vs full test gallery")
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        required_paths = [
            self.root,
            self.list_seq_train_path,
            self.list_seq_test_path,
            self.query_idx_path,
            self.info_test_path,
        ]
        for required_path in required_paths:
            if not osp.exists(required_path):
                raise RuntimeError("'{}' is not available".format(required_path))

    def _load_test_camids(self):
        info_test = np.load(self.info_test_path, allow_pickle=True)
        if info_test.ndim != 2 or info_test.shape[1] < 4:
            raise RuntimeError("Unexpected LS-VID test info format in '{}'".format(self.info_test_path))
        return info_test[:, 3].astype(np.int64) - 1

    def _read_sequence_lines(self, list_path):
        with open(list_path, 'r') as handle:
            return [line.strip() for line in handle if line.strip()]

    def _resolve_image_paths(self, seq_rel_path):
        seq_dir = osp.join(self.root, osp.dirname(seq_rel_path))
        seq_prefix = osp.basename(seq_rel_path)
        patterns = [
            osp.join(seq_dir, "{}_*.jpg".format(seq_prefix)),
            osp.join(seq_dir, "{}_*.png".format(seq_prefix)),
        ]
        img_paths = []
        for pattern in patterns:
            img_paths.extend(glob.glob(pattern))
        img_paths = sorted(img_paths)
        if not img_paths:
            raise RuntimeError("No frames found for LS-VID tracklet '{}'".format(seq_rel_path))
        return img_paths

    def _infer_camid_from_paths(self, img_paths):
        cam_tokens = {int(osp.basename(img_path).split('_')[2]) for img_path in img_paths}
        if len(cam_tokens) != 1:
            raise RuntimeError("LS-VID tracklet contains multiple camera ids: {}".format(img_paths[0]))
        return cam_tokens.pop() - 1

    def _process_sequence_list(self, list_path, relabel=False, min_seq_len=0, camid_lookup=None):
        lines = self._read_sequence_lines(list_path)
        pid_container = sorted({int(line.rsplit(' ', 1)[1]) for line in lines})
        pid2label = {pid: label for label, pid in enumerate(pid_container)} if relabel else None

        if camid_lookup is not None and len(camid_lookup) != len(lines):
            raise RuntimeError(
                "LS-VID camera metadata count {} does not match sequence list count {}".format(
                    len(camid_lookup), len(lines)
                )
            )

        tracklets = []
        num_imgs_per_tracklet = []
        for idx, line in enumerate(lines):
            seq_rel_path, pid_str = line.rsplit(' ', 1)
            pid = int(pid_str)
            img_paths = self._resolve_image_paths(seq_rel_path)
            if len(img_paths) < min_seq_len:
                continue

            camid = int(camid_lookup[idx]) if camid_lookup is not None else self._infer_camid_from_paths(img_paths)
            if relabel:
                pid = pid2label[pid]

            tracklets.append((tuple(img_paths), pid, camid))
            num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        num_pids = len(pid_container)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


def extract_tracklet_feat(
    img_paths,
    feat_extractor,
    chunk_size=32,
):
    all_frame_feats = []

    with torch.inference_mode():
        for start_idx in range(0, len(img_paths), chunk_size):
            chunk_paths = img_paths[start_idx:start_idx + chunk_size]

            img_tensors = prep_img_paths(list(chunk_paths))   # should end up on GPU
            chunk_feats = feat_extractor(img_tensors)         # [chunk, feat_dim]

            chunk_feats = chunk_feats.detach().cpu()
            all_frame_feats.append(chunk_feats)

            del img_tensors
            del chunk_feats

    frame_feats = torch.cat(all_frame_feats, dim=0)   # on CPU
    tracklet_feat = frame_feats.mean(dim=0)           # on CPU

    del frame_feats
    return tracklet_feat

def extract_all_feat(data, 
                     feat_extractor,
                     save_dir):
    all_tracklet_feat = {
        "features": [],
        "tracklet_ids": [],
        "cam_ids": []
    }

    with torch.inference_mode():
        for img_paths, tracklet_id, cam_id in tqdm(data):
            feat = extract_tracklet_feat(img_paths=img_paths,
                                         feat_extractor=feat_extractor)
            feat = feat.detach().cpu()

            all_tracklet_feat["features"].append(feat)
            all_tracklet_feat["tracklet_ids"].append(tracklet_id)
            all_tracklet_feat["cam_ids"].append(cam_id)

            del feat

    with open(os.path.join(save_dir, "all_tracklet_feat.pkl"), "wb") as f:
        pkl.dump(all_tracklet_feat, f)

    return all_tracklet_feat

def compute_dist_map(query_features, gallery_features):
    normalized_q_f = F.normalize(query_features, p=2, dim=1)
    normalized_g_f = F.normalize(gallery_features, p=2, dim=1)
    sim = normalized_q_f @ normalized_g_f.T
    dist = 1.0 - sim
    return dist

def evaluate_from_distmat(dist_mat, 
                          q_tracklet_ids, 
                          q_cam_ids,
                          g_tracklet_ids, 
                          g_cam_ids):
    num_q, num_g = dist_mat.shape

    if not torch.is_tensor(q_tracklet_ids):
        q_tracklet_ids = torch.tensor(q_tracklet_ids)
    if not torch.is_tensor(q_cam_ids):
        q_cam_ids = torch.tensor(q_cam_ids)
    if not torch.is_tensor(g_tracklet_ids):
        g_tracklet_ids = torch.tensor(g_tracklet_ids)
    if not torch.is_tensor(g_cam_ids):
        g_cam_ids = torch.tensor(g_cam_ids)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for i in range(num_q):
        q_t_id = q_tracklet_ids[i]
        q_c_id = q_cam_ids[i]

        order = torch.argsort(dist_mat[i])
        ordered_g_t_id = g_tracklet_ids[order]
        ordered_g_c_id = g_cam_ids[order]

        keep = ~((ordered_g_t_id == q_t_id) & (ordered_g_c_id == q_c_id))

        ordered_g_t_id = ordered_g_t_id[keep]
        matches = (ordered_g_t_id == q_t_id).int()

        if matches.sum() == 0:
            continue

        cmc = matches.cumsum(dim=0)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:5].float())

        # AP
        num_rel = matches.sum().item()
        precisions = matches.cumsum(dim=0).float() / (torch.arange(len(matches)).float() + 1)
        ap = (precisions * matches.float()).sum() / num_rel
        all_ap.append(ap.item())

        num_valid_q += 1

    all_cmc = torch.stack(all_cmc, dim=0)
    cmc = all_cmc.mean(dim=0)
    mAP = sum(all_ap) / len(all_ap)

    return {
        "rank1": cmc[0].item(),
        "rank5": cmc[4].item(),
        "mAP": mAP,
        "num_valid_queries": num_valid_q
    }

def main(results_dir=None,
         experiment_name=None):
    results_dir_root = "./results"

    if results_dir is None:
        _, transform_te = build_transforms(height=256, 
                                        width=128,
                                        transforms='random_flip')

        print("NO FEATURES FOUND, EXTRACTING FIRST")

        curr_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        results_subdir = f"{results_dir_root}/{curr_timestamp}_{EXPERIMENTS[experiment_name]['result_subdir_ext']}"

        os.makedirs(results_subdir, exist_ok=True)

        query_save_dir = f"{results_subdir}/query"
        gallery_save_dir = f"{results_subdir}/gallery"

        os.makedirs(query_save_dir, exist_ok=True)
        os.makedirs(gallery_save_dir, exist_ok=True)

        print(f"SAVING RESULTS TO: {query_save_dir} and {gallery_save_dir}")

        osnet_x1_0_feat_extractor = OSNet(
            1000,
            blocks=[OSBlock, OSBlock, OSBlock],
            layers=[2, 2, 2],
            channels=[64, 256, 384, 512],
            loss='softmax'
        ).to(torch.device("cuda"))

        print("Model size: {:.5f}M".format(sum(p.numel() for p in osnet_x1_0_feat_extractor.parameters()) / 1000000.0))
        
        # model init
        initialized_model = init_local_ckpt(model=osnet_x1_0_feat_extractor,
                                            ckpt_path=EXPERIMENTS[experiment_name]["ckpt_path"])
        initialized_model.eval()
        
        # data init
        lsvid = LSVID(root="../../../../data6/haidong/data/lsvid/",
                      min_seq_len=8)
        
        lsvid_query = lsvid.query
        lsvid_gallery = lsvid.gallery

        print("BEGIN EXTRACTING QUERY")
        _ = extract_all_feat(lsvid_query, initialized_model, query_save_dir)
        print("FINISHED EXTRACTING QUERY")

        print("BEGIN EXTRACTING GALLERY")
        _ = extract_all_feat(lsvid_gallery, initialized_model, gallery_save_dir)
        print("FINISHED EXTRACTING GALLERY")

        results_dir = results_subdir

    with open(os.path.join(results_dir, f"query/all_tracklet_feat.pkl"), "rb") as f:
        query_features = pkl.load(f)
    
    with open(os.path.join(results_dir, f"gallery/all_tracklet_feat.pkl"), "rb") as f:
        gallery_features = pkl.load(f)

    stacked_q_f = torch.stack(query_features["features"], dim=0)
    stacked_g_f = torch.stack(gallery_features["features"], dim=0)

    dist_mat = compute_dist_map(stacked_q_f,
                                stacked_g_f)

    results = evaluate_from_distmat(
        dist_mat,
        q_tracklet_ids=query_features["tracklet_ids"],
        q_cam_ids=query_features["cam_ids"],
        g_tracklet_ids=gallery_features["tracklet_ids"],
        g_cam_ids=gallery_features["cam_ids"]
    )

    print("Rank-1:", results["rank1"])
    print("Rank-5:", results["rank5"])
    print("mAP:", results["mAP"])

    _, transform_te = build_transforms(height=256, 
                                width=128,
                                transforms='random_flip')

    mars_dataset = Mars(root="../../../../data6/haidong/data/mars/",
                        transform=transform_te)

    mars_query = mars_dataset.query
    mars_gallery = mars_dataset.gallery

    visualize_ranked_results(distmat=dist_mat,
                             dataset=(mars_query, mars_gallery),
                             data_type="image",
                             save_dir="test")

if __name__ == "__main__":
    fire.Fire(main)

# first time (no features) cmds
# CUDA_VISIBLE_DEVICES=1 python -m src.eval_lsvid --experiment_name="osnet_duke" (~30 mins)
# CUDA_VISIBLE_DEVICES=2 python -m src.eval_lsvid --experiment_name="osnet_msmt" (~30 mins)

# evaluating (features exist!) cmds
# python -m src.eval_lsvid --results_dir="./results/2026_04_08_03_52_57_lsvid_market"
# python -m src.eval_lsvid --results_dir="./results/2026_04_08_03_59_42_lsvid_combine"
# python -m src.eval_mars --results_dir="./results/2026_04_07_22_58_46_market"
# 