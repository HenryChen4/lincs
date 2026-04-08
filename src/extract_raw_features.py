import pickle as pkl

qf_path = "./results/2026_04_07_23_25_00_market/query/all_tracklet_feat.pkl"
gf_path = "./results/2026_04_07_23_25_00_market/gallery/all_tracklet_feat.pkl"

with open(qf_path, "rb") as f:
    q_data = pkl.load(f)

with open(gf_path, "rb") as f:
    g_data = pkl.load(f)

with open("qf_output.txt", "w") as f:
    f.write(str(q_data['features']) + "\n")

with open("gf_output.txt", "w") as f:
    f.write(str(g_data['features']) + "\n")