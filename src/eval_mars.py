import torch
import json
import os
import pickle as pkl
import fire
import torch.nn.functional as F

from osnet.torchreid.data.datasets.video.mars import Mars # made some adjustments
from osnet.torchreid.data.transforms import RandomErasing, Random2DTranslation, RandomHorizontalFlip, RandomPatch, build_transforms
from osnet.torchreid.utils.feature_extractor import FeatureExtractor
from osnet.torchreid.utils.reidtools import visualize_ranked_results

from tqdm import tqdm
from datetime import datetime

EXPERIMENTS = {
    "osnet_market": {
        "result_subdir_ext": "market",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    },
    "osnet_duke": {
        "result_subdir_ext": "duke",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    },
    "osnet_msmt": {
        "result_subdir_ext": "msmt",
        "ckpt_path": "./pretrained_ckpt/osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    }
}

def extract_tracklet_feat(img_paths, 
                          feat_extractor):
    frame_feats = feat_extractor(list(img_paths))
    tracklet_feat = frame_feats.mean(dim=0)
    return tracklet_feat

def extract_all_feat(data, 
                     feat_extractor,
                     save_dir):
    all_tracklet_feat = {
        "features": [],
        "tracklet_ids": [],
        "cam_ids": []
    }
    
    for img_paths, tracklet_id, cam_id, _ in tqdm(data):
        feat = extract_tracklet_feat(img_paths=img_paths,
                                     feat_extractor=feat_extractor)

        all_tracklet_feat["features"].append(feat)
        all_tracklet_feat["tracklet_ids"].append(tracklet_id)
        all_tracklet_feat["cam_ids"].append(cam_id)

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

        osnet_x1_0_feat_extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=EXPERIMENTS[experiment_name]["ckpt_path"],
            device="cuda"
        )
        
        mars_dataset = Mars(root="../../../../data6/haidong/data/mars/",
                            transform=transform_te)

        mars_query = mars_dataset.query
        mars_gallery = mars_dataset.gallery

        print("BEGIN EXTRACTING QUERY")
        _ = extract_all_feat(mars_query, osnet_x1_0_feat_extractor, query_save_dir)
        print("FINISHED EXTRACTING QUERY")

        print("BEGIN EXTRACTING GALLERY")
        _ = extract_all_feat(mars_gallery, osnet_x1_0_feat_extractor, gallery_save_dir)
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

    # _, transform_te = build_transforms(height=256, 
    #                             width=128,
    #                             transforms='random_flip')

    # mars_dataset = Mars(root="../../../../data6/haidong/data/mars/",
    #                     transform=transform_te)

    # mars_query = mars_dataset.query
    # mars_gallery = mars_dataset.gallery

    # visualize_ranked_results(distmat=dist_mat,
    #                          dataset=(mars_query, mars_gallery),
    #                          data_type="image",
    #                          save_dir="test")

if __name__ == "__main__":
    fire.Fire(main)

# first time (no features) cmds
# CUDA_VISIBLE_DEVICES=1 python -m src.eval_mars --experiment_name="osnet_market" (~30 mins)
# CUDA_VISIBLE_DEVICES=2 python -m src.eval_mars --experiment_name="osnet_duke"
# CUDA_VISIBLE_DEVICES=3 python -m src.eval_mars --experiment_name="osnet_msmt"

# evaluating (features exist!) cmds
# python -m src.eval_mars --results_dir="./results/2026_04_06_04_14_26_market"
# python -m src.eval_mars --results_dir="./results/2026_04_06_04_19_47_duke"
# python -m src.eval_mars --results_dir="./results/2026_04_06_04_20_21_msmt"