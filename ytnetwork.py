


#imports
#note, run in current submission has an error since my final run (after I generated all graphs/values for the report)
#exceeded my api quota . With your own API key, it will run.
import re
import json
import time
from collections import defaultdict
from googleapiclient.discovery import build
from tqdm import tqdm
import pandas as pd
import numpy as np

#setup API Key:
#NOTE: API Key has been removed from final submission for security purposes. To run the code yourself,
#      generate your own Youtube Data v3 API key through the Google Developers Console

API_KEY = ''  #enter key here
youtube = build("youtube", "v3", developerKey=API_KEY)

#helper functions for playlist parsing
def extract_playlist_id(url):
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def get_all_videos_from_playlist(playlist_id):
    video_ids = []
    next_page = None
    while True:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page
        ).execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page = response.get('nextPageToken')
        if not next_page:
            break
    return video_ids

def get_video_metadata(video_ids):
    meta = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        response = youtube.videos().list(
            part="snippet",
            id=','.join(batch)
        ).execute()

        for item in response['items']:
            vid = item['id']
            snippet = item['snippet']
            meta[vid] = {
                "title": snippet['title'],
                "channel": snippet['channelTitle']
            }
    return meta

def get_commenters(video_id):
    commenters = set()
    total_comments = 0
    try:
        next_page = None
        while True:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                author = item["snippet"]["topLevelComment"]["snippet"].get("authorDisplayName")
                if author:
                    commenters.add(author)
                total_comments += 1

            next_page = response.get("nextPageToken")
            if not next_page:
                break
    except Exception as e:
        print(f"[!] Error on video {video_id}: {e}")
    return commenters, total_comments

#playlists of videos to be used as dataset

playlist_urls = [
     'https://www.youtube.com/watch?v=wkPlj4fE1Ng&list=PLgxL0tm8FuoeCSN0KgPW4dq2LnzmrLBdZ',
     'https://www.youtube.com/watch?v=CQBOA061ugE&list=PLt6oQcvfST4PT8csOpMhO6JbasOotBCet'
]

#collect video metadata
all_video_metadata = {}
video_creator_dict_raw = defaultdict(list)

for url in playlist_urls:
    print(f"Processing playlist: {url}")
    playlist_id = extract_playlist_id(url)
    video_ids = get_all_videos_from_playlist(playlist_id)
    metadata = get_video_metadata(video_ids)

    for vid in video_ids:
        if vid not in metadata:
            continue
        channel = metadata[vid]['channel']
        title = metadata[vid]['title']

        video_creator_dict_raw[channel].append({
            "id": vid,
            "title": title
        })
        all_video_metadata[vid] = {
            "title": title,
            "channel": channel
        }

print(f"\n Total channels before filtering: {len(video_creator_dict_raw)}")

#filter out channels with less than ten videos on the playlist
video_creator_dict = {
    channel: videos
    for channel, videos in video_creator_dict_raw.items()
    if len(videos) >= 10
}
print(f"Filtered to {len(video_creator_dict)} channels with ≥10 videos each")

#get comments from videos
all_commenters_by_video = {}
total_comments_all_videos = 0

for channel, videos in video_creator_dict.items():
    for video in tqdm(videos, desc=f"Fetching comments for {channel}"):
        vid = video["id"]
        commenters, comment_count = get_commenters(vid)
        all_commenters_by_video[vid] = commenters
        total_comments_all_videos += comment_count

        print(f"{channel} - {vid}: {len(commenters)} unique commenters, {comment_count} total comments")
        time.sleep(0.1)

print(f"\n Total comments across all videos: {total_comments_all_videos}")

#viewer overlap matrix using jaccard
video_ids = list(all_commenters_by_video.keys())
sim_matrix = pd.DataFrame(np.zeros((len(video_ids), len(video_ids))), index=video_ids, columns=video_ids)

for i, v1 in enumerate(video_ids):
    commenters1 = all_commenters_by_video.get(v1, set())

    for j, v2 in enumerate(video_ids):
        if i >= j:
            continue

        commenters2 = all_commenters_by_video.get(v2, set())
        intersection = len(commenters1 & commenters2)
        union = len(commenters1 | commenters2)
        sim = intersection / union if union > 0 else 0

        sim_matrix.iloc[i, j] = sim
        sim_matrix.iloc[j, i] = sim

#map colors to channels
import plotly.colors as pc
glasbey_colors = pc.qualitative.Alphabet + pc.qualitative.Light24 + pc.qualitative.Dark24
all_channels = sorted(set(video_creator_dict.keys()))

CHANNEL_COLOR_MAP = {
    ch: glasbey_colors[i % len(glasbey_colors)] for i, ch in enumerate(all_channels)
}

node_colors = [
    CHANNEL_COLOR_MAP[active_graph.nodes[node]["channel"]]
    for node in active_graph.nodes()
]

import networkx as nx

#build graph from dataset
#convert dictionary into flat list for iterative purposes
all_videos = []
for channel, videos in video_creator_dict.items():
    for video in videos:
        all_videos.append({
            "id": video["id"],
            "title": video["title"],
            "channel": channel
        })

print(f"Total videos used: {len(all_videos)}")

graph = nx.Graph()

#add video nodes with metadata
for video in all_videos:
    vid = video["id"]
    graph.add_node(vid, title=video["title"], channel=video["channel"])

#add edges based on jaccard similarity above a certain threshold
min_jaccard_threshold = 0.01

for i in range(len(all_videos)):
    for j in range(i + 1, len(all_videos)):
        v1 = all_videos[i]["id"]
        v2 = all_videos[j]["id"]

        commenters1 = all_commenters_by_video.get(v1, set())
        commenters2 = all_commenters_by_video.get(v2, set())

        union = commenters1 | commenters2
        if not union:
            continue

        intersection = commenters1 & commenters2
        jaccard = len(intersection) / len(union)

        if jaccard >= min_jaccard_threshold:
            graph.add_edge(
                v1,
                v2,
                weight=jaccard,
                same_channel=(graph.nodes[v1]["channel"] == graph.nodes[v2]["channel"])
            )

print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")



import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import plotly.colors as pc

active_graph = graph.copy()

#get colors for channels
#legend for node channels/colors
present_channels = sorted(set(nx.get_node_attributes(active_graph, "channel").values()))
legend_patches = [
    mpatches.Patch(color=CHANNEL_COLOR_MAP[ch], label=ch)
    for ch in present_channels
]

#normalize jaccard values to represent as edge widths
edge_weights = [d["weight"] for _, _, d in active_graph.edges(data=True)]
min_w, max_w = min(edge_weights), max(edge_weights)
normalized_weights = [
    1 + 4 * (w - min_w) / (max_w - min_w) if max_w > min_w else 1
    for w in edge_weights
]

all_edges = list(active_graph.edges(data=True))
edge_width_map = {
    (u, v): w for (u, v, d), w in zip(all_edges, normalized_weights)
}

#draw graph
plt.figure(figsize=(18, 18))
pos = nx.spring_layout(active_graph, seed=42)

nx.draw_networkx_nodes(
    active_graph,
    pos,
    node_size=30,
    node_color=node_colors
)

intra_edges = [(u, v) for u, v, d in active_graph.edges(data=True) if d.get("same_channel")]
filtered_intra_edges = [(u, v) for u, v in graph.edges() if u != v]
inter_edges = [(u, v) for u, v, d in active_graph.edges(data=True) if not d.get("same_channel")]
nx.draw_networkx_edges(
    active_graph,
    pos,
    edgelist=intra_edges,
    alpha=0.08,
    width=[edge_width_map[(u, v)] for u, v in filtered_intra_edges],
    edge_color='gray'
)
nx.draw_networkx_edges(
    active_graph,
    pos,
    edgelist=inter_edges,
    alpha=0.4,
    width=[edge_width_map[(u, v)] for u, v in inter_edges],
    edge_color='black'
)

#add legend
plt.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    title="Channel",
    fontsize=8,
    borderaxespad=0
)

plt.title("Communities Among Iceberg YouTubers", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()



from collections import defaultdict
import matplotlib.cm as cm
import networkx as nx
import community.community_louvain as community_louvain
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.colors as mcolors

active_graph.remove_edges_from(nx.selfloop_edges(active_graph))

#recursive louvain community detection
def recursive_louvain_assign(graph, max_depth=3, current_depth=1, base_label=()):

    partition = community_louvain.best_partition(graph, weight="weight")
    for node, comm in partition.items():
        graph.nodes[node]['community'] = base_label + (comm,)

    if current_depth < max_depth:
        comm_to_nodes = defaultdict(list)
        for node, comm in partition.items():
            comm_to_nodes[comm].append(node)

        for comm, nodes in comm_to_nodes.items():
            subgraph = graph.subgraph(nodes)
            recursive_louvain_assign(
                subgraph, max_depth=max_depth,
                current_depth=current_depth + 1,
                base_label=base_label + (comm,)
            )


def generate_distinct_colors(n):
    cmap = cm.get_cmap('gist_ncar', n)
    return [cmap(i) for i in range(n)]

#plot the communities detected

def plot_louvain_communities_by_depth(graph, depth_level=0, seed=42, figsize=(18, 18), title=None):
    depth_labels = {}
    for node in graph.nodes():
        label_tuple = graph.nodes[node].get('community', ())
        label_at_depth = label_tuple[:depth_level + 1] if len(label_tuple) > depth_level else (-1,)
        depth_labels[node] = label_at_depth

    unique_comms = sorted(set(depth_labels.values()))
    n_comms = len(unique_comms)
    comm_to_color = {
        comm_id: color for comm_id, color in zip(unique_comms, generate_distinct_colors(n_comms))
    }

    node_colors = [comm_to_color[depth_labels[n]] for n in graph.nodes()]
    pos = nx.spring_layout(graph, seed=seed)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=30)
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=graph.edges(),
        alpha=0.2,
        width=[1 + 3 * d.get("weight", 1.0) for _, _, d in graph.edges(data=True)],
        edge_color='black'
    )
    plt.title(title or f"Louvain Communities (Depth {depth_level}) — {n_comms} communities", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_louvain_communities_by_depth(active_graph, depth_level=0, title="Louvain Communities (Depth 0)")
plot_louvain_communities_by_depth(active_graph, depth_level=1, title="Louvain Communities (Depth 1)")
plot_louvain_communities_by_depth(active_graph, depth_level=2, title="Louvain Communities (Depth 2)")

import math
from collections import defaultdict
from scipy.stats import hypergeom

def compute_surprise(graph, community_dict):
    total_edges = graph.number_of_edges()
    total_possible_edges = len(graph) * (len(graph) - 1) // 2

    #count intra-community links and possible intra-community pairs
    intra_edges = 0
    intra_possible_edges = 0

    for nodes in community_dict.values():
        subg = graph.subgraph(nodes)
        intra_edges += subg.number_of_edges()
        n = len(nodes)
        intra_possible_edges += n * (n - 1) // 2

    F = total_possible_edges
    F_in = intra_possible_edges
    L = total_edges
    k = intra_edges

    log_p = hypergeom.logsf(k - 1, F, F_in, L)
    surprise_score = -log_p

    return surprise_score

def get_community_dict_by_depth(graph, depth):
    comm_dict = defaultdict(list)
    for node in graph.nodes():
        label = graph.nodes[node].get("community", ())
        comm_id = label[depth] if len(label) > depth else -1
        comm_dict[comm_id].append(node)
    return comm_dict

for depth in range(3):
    community_dict = get_community_dict_by_depth(active_graph, depth)
    surprise = compute_surprise(active_graph, community_dict)
    print(f"Louvain Depth {depth} — Surprise Score: {surprise:.4f}")

#generate null partitions
import random

def generate_random_partition(graph, community_dict):

    nodes = list(graph.nodes())
    random.shuffle(nodes)

    community_sizes = [len(nodes_list) for nodes_list in community_dict.values()]

    new_partition = {}
    idx = 0
    for i, size in enumerate(community_sizes):
        new_partition[i] = nodes[idx : idx + size]
        idx += size

    return new_partition

def compute_random_surprise_scores(graph, community_dict, num_samples=200):
    random_scores = []

    for _ in range(num_samples):
        rand_partition = generate_random_partition(graph, community_dict)
        S_rand = compute_surprise(graph, rand_partition)
        random_scores.append(S_rand)

    return random_scores

#calculate z-scores

for depth in range(3):
    print(f"\n=== Depth {depth} ===")

    community_dict = get_community_dict_by_depth(active_graph, depth)
    S_louvain = compute_surprise(active_graph, community_dict)

    print(f"Louvain Surprise: {S_louvain:.4f}")


    rand_scores = compute_random_surprise_scores(active_graph, community_dict, num_samples=200)
    mean_rand = np.mean(rand_scores)
    max_rand = np.max(rand_scores)
    min_rand = np.min(rand_scores)
    std_rand = np.std(rand_scores)

    print("Random Baseline:")
    print(f"  Mean Surprise: {mean_rand:.4f}")
    print(f"  Std Surprise : {std_rand:.4f}")
    print(f"  Max Surprise : {max_rand:.4f}")
    print(f"  Min Surprise : {min_rand:.4f}")

    z = (S_louvain - mean_rand) / (std_rand + 1e-9)
    print(f"  Z-score vs Random: {z:.4f}")

#compute centrality measures
degree_centrality = nx.degree_centrality(active_graph)
betweenness_centrality = nx.betweenness_centrality(active_graph, weight='weight', normalized=True)

centrality_df = pd.DataFrame({
    "video_id": list(active_graph.nodes()),
    "title": [active_graph.nodes[n]["title"] for n in active_graph.nodes()],
    "channel": [active_graph.nodes[n]["channel"] for n in active_graph.nodes()],
    "degree_centrality": [degree_centrality[n] for n in active_graph.nodes()],
    "betweenness_centrality": [betweenness_centrality[n] for n in active_graph.nodes()],
})

top_degree = centrality_df.sort_values("degree_centrality", ascending=False).head(10)
top_betweenness = centrality_df.sort_values("betweenness_centrality", ascending=False).head(10)

print("Top Videos by Degree Centrality:")
display(top_degree[["title", "channel", "degree_centrality"]])

print("\nTop Videos by Betweenness Centrality:")
display(top_betweenness[["title", "channel", "betweenness_centrality"]])

import matplotlib.patches as mpatches
import seaborn as sns

top_n = 10
top_degree_nodes = (
    centrality_df.sort_values("degree_centrality", ascending=False)
    .head(top_n)["video_id"]
    .tolist()
)
top_betweenness_nodes = (
    centrality_df.sort_values("betweenness_centrality", ascending=False)
    .head(top_n)["video_id"]
    .tolist()
)

central_nodes = list(set(top_degree_nodes + top_betweenness_nodes))

print(f"Total highlighted central nodes: {len(central_nodes)}")

palette = sns.color_palette("tab20", len(central_nodes))
node_color_map = {node: palette[i] for i, node in enumerate(central_nodes)}

pos = nx.spring_layout(active_graph, seed=42)

plt.figure(figsize=(22, 22))

nx.draw_networkx_edges(
    active_graph,
    pos,
    edge_color="lightgray",
    width=0.4,
    alpha=0.15
)

nx.draw_networkx_nodes(
    active_graph,
    pos,
    node_size=18,
    node_color="gray",
    alpha=0.20
)


nx.draw_networkx_nodes(
    active_graph,
    pos,
    nodelist=central_nodes,
    node_size=420,
    node_color=[node_color_map[n] for n in central_nodes],
    edgecolors="black",
    linewidths=1.2,
    alpha=0.95
)


central_edges = []
for n in central_nodes:
    central_edges.extend([(n, nbr) for nbr in active_graph.neighbors(n)])

nx.draw_networkx_edges(
    active_graph,
    pos,
    edgelist=central_edges,
    width=2.0,
    alpha=0.55,
    edge_color=[node_color_map[n] for (n, _) in central_edges]
)

central_labels = {
    n: active_graph.nodes[n]["title"][:22] + "..." if len(active_graph.nodes[n]["title"]) > 22
    else active_graph.nodes[n]["title"]
    for n in central_nodes
}

nx.draw_networkx_labels(
    active_graph,
    pos,
    labels=central_labels,
    font_size=9,
    font_color="black",
    font_weight="bold"
)

legend_handles = []

for n in central_nodes:
    color = node_color_map[n]
    title = active_graph.nodes[n]["title"]
    short_title = title[:45] + "..." if len(title) > 45 else title

    handle = mpatches.Patch(color=color, label=short_title)
    legend_handles.append(handle)

plt.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=10,
    title="Central Videos (Hubs & Bridges)",
    title_fontsize=12
)

plt.title("Central Hub & Bridge Videos", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.show()

#export top 5 communities at each depth
import pandas as pd
from collections import defaultdict

def get_community_dict_by_depth(graph, depth):
    comm_dict = defaultdict(list)
    for node in graph.nodes():
        label = graph.nodes[node].get("community", ())
        comm_id = label[depth] if len(label) > depth else -1
        comm_dict[comm_id].append(node)
    return comm_dict


def build_community_df(graph, communities, depth):
    rows = []

    for comm_id, nodes in communities.items():
        for n in nodes:
            rows.append({
                "community_id": comm_id,
                "title": graph.nodes[n]["title"],
                "channel": graph.nodes[n]["channel"],
                "genre": ""
            })

    df = pd.DataFrame(rows)
    return df


depths_to_export = [0, 1, 2]

for depth in depths_to_export:

    comm_dict = get_community_dict_by_depth(active_graph, depth)

    if -1 in comm_dict:
        del comm_dict[-1]

    sorted_comms = sorted(comm_dict.items(), key=lambda x: len(x[1]), reverse=True)

    top5 = dict(sorted_comms[:5])

    df_export = build_community_df(active_graph, top5, depth)

    out_name = f"louvain_depth{depth}_top5_communities.csv"
    df_export.to_csv(out_name, index=False, encoding="utf-8")

    print(f"Exported: {out_name}  (rows={len(df_export)})")





