"""
Multimodal CLIP + BLIP System
Image-Text Alignment, Retrieval, Zero-Shot Classification & Visual Q&A

Mirrors the notebook pipeline but uses a different dataset
(nature, vehicles, sports, food, architecture).

Install:
    pip install transformers torch torchvision Pillow matplotlib requests
    pip install open-clip-torch
"""

import os
import json
import warnings
import textwrap
from io import BytesIO

import requests
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)

from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
print("\nLoading CLIP (openai/clip-vit-base-patch32)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading BLIP captioning (Salesforce/blip-image-captioning-base)...")
blip_cap_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
blip_cap_model.eval()

print("Loading BLIP VQA (Salesforce/blip-vqa-base)...")
blip_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(DEVICE)
blip_vqa_model.eval()


# ─────────────────────────────────────────────
# Image loader
# ─────────────────────────────────────────────
def load_image_from_url(url: str) -> Image.Image:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"   Could not load {url}: {e}")
        img = Image.new("RGB", (224, 224), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 224, 224], outline=(150, 150, 150), width=3)
        draw.text((30, 95), "Image", fill=(80, 80, 80))
        draw.text((18, 115), "Unavailable", fill=(80, 80, 80))
        return img


# ─────────────────────────────────────────────
# Dataset (different from the notebook)
# ─────────────────────────────────────────────
IMAGE_URLS = [
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=320",   # mountain landscape
    "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=320",   # beach sunset
    "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=320",   # robot / tech
    "https://images.unsplash.com/photo-1542362567-b07e54358753?w=320",       # red sports car
    "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=320",    # airplane in sky
    "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=320",   # basketball game
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=320",    # gourmet burger
    "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=320",    # city skyline at night
]

CAPTIONS = [
    "A snow-capped mountain landscape under a clear sky",
    "A peaceful beach during a colourful sunset",
    "A humanoid robot showcasing modern technology",
    "A bright red sports car parked on a road",
    "A passenger airplane flying through the sky",
    "Players competing in an indoor basketball match",
    "A delicious gourmet burger served on a plate",
    "A city skyline glowing with lights at night",
]

IMAGE_LABELS = [
    "mountain", "beach sunset", "robot", "sports car",
    "airplane", "basketball", "burger", "city skyline",
]

print("\nDownloading sample images from Unsplash...")
images = [load_image_from_url(url) for url in IMAGE_URLS]
print(f"   Loaded {len(images)} images.\n")


# ─────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────
def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract tensor from {type(output)}")


@torch.no_grad()
def get_image_embeddings(pil_images: list) -> torch.Tensor:
    inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    feats = clip_model.get_image_features(**inputs)
    feats = _to_tensor(feats)
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def get_text_embeddings(texts: list) -> torch.Tensor:
    inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    feats = clip_model.get_text_features(**inputs)
    feats = _to_tensor(feats)
    return F.normalize(feats, dim=-1)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.T


# ─────────────────────────────────────────────
# Pre-compute embeddings
# ─────────────────────────────────────────────
print("Computing image embeddings...")
image_embeddings = get_image_embeddings(images)

print("Computing caption embeddings...")
caption_embeddings = get_text_embeddings(CAPTIONS)

sim_matrix = cosine_similarity_matrix(image_embeddings, caption_embeddings)
print(f"   Similarity matrix shape: {sim_matrix.shape}")


# ─────────────────────────────────────────────
# Similarity matrix plot
# ─────────────────────────────────────────────
def plot_similarity_matrix(sim, row_labels, col_labels,
                           title="CLIP Cosine Similarity Matrix"):
    data = sim.cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    wrapped = [textwrap.fill(c, 18) for c in col_labels]
    ax.set_xticklabels(wrapped, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Captions", fontsize=10)
    ax.set_ylabel("Images", fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="black" if data[i, j] < 0.7 else "white",
            )
    plt.tight_layout()
    fpath = out_path("similarity_matrix.png")
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved -> {fpath}")


plot_similarity_matrix(sim_matrix, IMAGE_LABELS, CAPTIONS)


# ─────────────────────────────────────────────
# Task 1: Image -> Top-k captions
# ─────────────────────────────────────────────
def image_to_top_k_captions(query_image, caption_pool, k=3):
    img_emb = get_image_embeddings([query_image])
    cap_embs = get_text_embeddings(caption_pool)
    scores = (img_emb @ cap_embs.T).squeeze(0)
    topk = scores.topk(min(k, len(caption_pool)))
    return [
        {"caption": caption_pool[i], "score": scores[i].item()}
        for i in topk.indices.tolist()
    ]


def display_image_to_captions(image, results, image_label="Query Image"):
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    axes[0].imshow(image)
    axes[0].set_title(f"{image_label}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    captions = [r["caption"] for r in results]
    scores = [r["score"] for r in results]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"][: len(results)]

    bars = axes[1].barh(
        range(len(results)), scores, color=colors, edgecolor="white", height=0.55
    )
    axes[1].set_yticks(range(len(results)))
    axes[1].set_yticklabels([textwrap.fill(c, 38) for c in captions], fontsize=9)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Cosine Similarity", fontsize=10)
    axes[1].set_title("Top-k Matched Captions", fontsize=11, fontweight="bold")
    axes[1].invert_yaxis()

    for bar, score in zip(bars, scores):
        axes[1].text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(
        out_path(f"image_to_captions_{image_label.replace(' ', '_')}.png"),
        dpi=120, bbox_inches="tight",
    )
    plt.close()


print("\n" + "=" * 60)
print("  TASK 1 - Image -> Top-k Captions")
print("=" * 60)
for idx in [0, 3, 5]:  # mountain, sports car, basketball
    results = image_to_top_k_captions(images[idx], CAPTIONS, k=3)
    print(f"\n[{IMAGE_LABELS[idx]}]")
    for rank, r in enumerate(results, 1):
        print(f"   #{rank}  ({r['score']:.3f})  {r['caption']}")
    display_image_to_captions(images[idx], results, IMAGE_LABELS[idx])


# ─────────────────────────────────────────────
# Task 2: Text -> Top-k images
# ─────────────────────────────────────────────
def text_to_top_k_images(query_text, image_pool, image_pool_embs, k=3):
    txt_emb = get_text_embeddings([query_text])
    scores = (txt_emb @ image_pool_embs.T).squeeze(0)
    topk = scores.topk(min(k, len(image_pool)))
    return [
        {"image": image_pool[i], "score": scores[i].item(), "index": i}
        for i in topk.indices.tolist()
    ]


def display_text_to_images(query, results):
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4.5))
    axes[0].text(
        0.5, 0.5, f'Query:\n\n"{query}"',
        ha="center", va="center", fontsize=11, wrap=True,
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#dfe6e9", alpha=0.9),
    )
    axes[0].axis("off")
    axes[0].set_title("Text Query", fontweight="bold")

    for rank, (ax, res) in enumerate(zip(axes[1:], results), 1):
        ax.imshow(res["image"])
        ax.set_title(f"#{rank}  score={res['score']:.3f}",
                     fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Text -> Image Retrieval", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = out_path(f"text_retrieval_{query[:20].replace(' ', '_')}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()


print("\n" + "=" * 60)
print("  TASK 2 - Text -> Image Retrieval")
print("=" * 60)
QUERIES = [
    "snowy mountain peak",
    "sunset over the ocean",
    "a fast red car",
    "a tasty burger meal",
    "lights of a big city at night",
]
for q in QUERIES:
    results = text_to_top_k_images(q, images, image_embeddings, k=3)
    print(f'\nQuery: "{q}"')
    for rank, r in enumerate(results, 1):
        print(
            f"   #{rank}  ({r['score']:.3f})  Image[{r['index']}] -> "
            f"{IMAGE_LABELS[r['index']]}"
        )
    display_text_to_images(q, results)


# ─────────────────────────────────────────────
# Task 3: Zero-shot classification
# ─────────────────────────────────────────────
def zero_shot_classify(query_image, class_labels,
                       prompt_template="a photo of a {}"):
    prompts = [prompt_template.format(c) for c in class_labels]
    img_emb = get_image_embeddings([query_image])
    txt_embs = get_text_embeddings(prompts)
    logits = (100.0 * img_emb @ txt_embs.T).squeeze(0)
    probs = logits.softmax(dim=-1).cpu().numpy()
    ranked = sorted(
        zip(class_labels, probs.tolist()), key=lambda x: x[1], reverse=True
    )
    return {
        "predicted": ranked[0][0],
        "confidence": ranked[0][1],
        "all": ranked,
    }


def display_classification(image, result, title=""):
    labels = [r[0] for r in result["all"]]
    probs = [r[1] for r in result["all"]]
    colors = [
        "#2ecc71" if l == result["predicted"] else "#95a5a6" for l in labels
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    axes[0].imshow(image)
    axes[0].set_title(
        f"{title}\nPredicted: {result['predicted']}\n"
        f"Confidence: {result['confidence']:.1%}",
        fontsize=10, fontweight="bold",
    )
    axes[0].axis("off")

    bars = axes[1].barh(
        range(len(labels)), probs, color=colors, edgecolor="white"
    )
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability", fontsize=10)
    axes[1].set_title("Zero-Shot Class Probabilities",
                      fontsize=11, fontweight="bold")
    axes[1].invert_yaxis()
    for bar, p in zip(bars, probs):
        axes[1].text(
            p + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}", va="center", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(
        out_path(f"zeroshot_{title.replace(' ', '_')}.png"),
        dpi=120, bbox_inches="tight",
    )
    plt.close()


CLASS_VOCAB = [
    "mountain", "beach", "robot", "car", "airplane",
    "basketball", "burger", "city", "forest", "river",
]

print("\n" + "=" * 60)
print("  TASK 3 - Zero-Shot Classification")
print("=" * 60)
for idx in range(len(images)):
    res = zero_shot_classify(images[idx], CLASS_VOCAB)
    print(
        f"\n[{IMAGE_LABELS[idx]}] -> predicted: {res['predicted']}  "
        f"({res['confidence']:.1%})"
    )
    display_classification(images[idx], res, IMAGE_LABELS[idx])


# ─────────────────────────────────────────────
# Task 4: Visual Q&A
# ─────────────────────────────────────────────
@torch.no_grad()
def visual_qa(image, question):
    inputs = blip_vqa_processor(image, question, return_tensors="pt").to(DEVICE)
    out = blip_vqa_model.generate(**inputs, max_new_tokens=50)
    return blip_vqa_processor.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def generate_caption(image, conditional_text=""):
    if conditional_text:
        inputs = blip_cap_processor(
            image, conditional_text, return_tensors="pt"
        ).to(DEVICE)
    else:
        inputs = blip_cap_processor(image, return_tensors="pt").to(DEVICE)
    out = blip_cap_model.generate(**inputs, max_new_tokens=60)
    return blip_cap_processor.decode(out[0], skip_special_tokens=True)


VQA_PAIRS = [
    (0, "What kind of landscape is shown?"),
    (1, "What time of day is it?"),
    (3, "What colour is the car?"),
    (4, "What is flying in the sky?"),
    (5, "What sport is being played?"),
    (6, "What food is on the plate?"),
    (7, "Is this a city or a village?"),
]

print("\n" + "=" * 60)
print("  TASK 4 - Visual Question Answering (BLIP-VQA)")
print("=" * 60)

vqa_records = []
for img_idx, question in VQA_PAIRS:
    answer = visual_qa(images[img_idx], question)
    vqa_records.append((img_idx, question, answer))
    print(f"\nImage: [{IMAGE_LABELS[img_idx]}]")
    print(f"   Q: {question}")
    print(f"   A: {answer}")


def display_vqa_results(records, all_images, all_labels):
    n = len(records)
    fig, axes = plt.subplots(
        n, 2, figsize=(11, 3.2 * n), gridspec_kw={"width_ratios": [1, 2]}
    )
    if n == 1:
        axes = [axes]
    for ax_row, (img_idx, question, answer) in zip(axes, records):
        ax_row[0].imshow(all_images[img_idx])
        ax_row[0].set_title(all_labels[img_idx], fontsize=9, fontweight="bold")
        ax_row[0].axis("off")
        ax_row[1].text(
            0.05, 0.65, f"Q: {question}",
            fontsize=10, fontweight="bold",
            transform=ax_row[1].transAxes, wrap=True,
        )
        ax_row[1].text(
            0.05, 0.28, f"A: {answer}",
            fontsize=10, color="#2c3e50",
            transform=ax_row[1].transAxes, wrap=True,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#d5f5e3", alpha=0.85),
        )
        ax_row[1].axis("off")
    plt.suptitle(
        "Visual Question Answering (BLIP-VQA)",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path("vqa_results.png"), dpi=120, bbox_inches="tight")
    plt.close()


display_vqa_results(vqa_records, images, IMAGE_LABELS)


# ─────────────────────────────────────────────
# Task 5: Caption generation & refinement
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 5 - Caption Generation & Refinement (BLIP)")
print("=" * 60)

caption_results = []
for idx in range(len(images)):
    base_caption = generate_caption(images[idx])
    refined_caption = generate_caption(
        images[idx], conditional_text="a detailed photo of"
    )
    caption_results.append((idx, base_caption, refined_caption))
    print(f"\n[{IMAGE_LABELS[idx]}]")
    print(f"   Base    : {base_caption}")
    print(f"   Refined : {refined_caption}")


def display_caption_comparison(results, all_images, all_labels):
    n = len(results)
    fig, axes = plt.subplots(
        n, 2, figsize=(13, 3.2 * n), gridspec_kw={"width_ratios": [1, 2.2]}
    )
    if n == 1:
        axes = [axes]
    for ax_row, (idx, base, refined) in zip(axes, results):
        ax_row[0].imshow(all_images[idx])
        ax_row[0].set_title(all_labels[idx], fontsize=9, fontweight="bold")
        ax_row[0].axis("off")

        ax_row[1].text(
            0.03, 0.78, "Base Caption:",
            fontsize=9, fontweight="bold",
            transform=ax_row[1].transAxes,
        )
        ax_row[1].text(
            0.03, 0.60, textwrap.fill(base, 68),
            fontsize=9, transform=ax_row[1].transAxes,
            bbox=dict(boxstyle="round", facecolor="#fde8d8", alpha=0.85),
        )
        ax_row[1].text(
            0.03, 0.38, "Refined Caption:",
            fontsize=9, fontweight="bold",
            transform=ax_row[1].transAxes,
        )
        ax_row[1].text(
            0.03, 0.18, textwrap.fill(refined, 68),
            fontsize=9, transform=ax_row[1].transAxes,
            bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.85),
        )
        ax_row[1].axis("off")

    plt.suptitle(
        "Caption Generation & Refinement (BLIP)",
        fontsize=13, fontweight="bold", y=1.003,
    )
    plt.tight_layout()
    plt.savefig(out_path("caption_refinement.png"), dpi=120, bbox_inches="tight")
    plt.close()


display_caption_comparison(caption_results, images, IMAGE_LABELS)


# ─────────────────────────────────────────────
# Full pipeline on a custom image
# ─────────────────────────────────────────────
def full_pipeline(image_url, caption_pool, image_pool, image_pool_embs,
                  class_vocab, questions, top_k=3):
    print("\n" + "=" * 70)
    print("  FULL MULTIMODAL PIPELINE")
    print("=" * 70)

    img = load_image_from_url(image_url)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title("Query Image", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path("pipeline_query.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print("\n-- Step 1: Image -> Top-k Captions --")
    cap_results = image_to_top_k_captions(img, caption_pool, k=top_k)
    for rank, r in enumerate(cap_results, 1):
        print(f"   #{rank} ({r['score']:.3f}) {r['caption']}")
    display_image_to_captions(img, cap_results, "Pipeline Query")

    print("\n-- Step 2: Caption Generation & Refinement (BLIP) --")
    base_cap = generate_caption(img)
    refined_cap = generate_caption(img, "a detailed photo of")
    print(f"   Base    : {base_cap}")
    print(f"   Refined : {refined_cap}")

    print("\n-- Step 3: Zero-Shot Classification --")
    clf = zero_shot_classify(img, class_vocab)
    print(f"   Predicted: {clf['predicted']}  ({clf['confidence']:.1%})")
    for label, prob in clf["all"][:5]:
        bar = "#" * int(prob * 30)
        print(f"      {label:<12} {bar} {prob:.1%}")
    display_classification(img, clf, "Pipeline Query")

    print("\n-- Step 4: Visual Q&A --")
    qa_pairs = []
    for q in questions:
        ans = visual_qa(img, q)
        qa_pairs.append((0, q, ans))
        print(f"   Q: {q}")
        print(f"   A: {ans}")
    display_vqa_results(
        [(0, q, a) for _, q, a in qa_pairs],
        [img] * len(questions),
        ["Query"] * len(questions),
    )

    print("\n-- Step 5: Text -> Image Retrieval using Generated Caption --")
    retrieval_results = text_to_top_k_images(
        refined_cap, image_pool, image_pool_embs, k=top_k
    )
    print(f'   Query: "{refined_cap}"')
    for rank, r in enumerate(retrieval_results, 1):
        print(
            f"   #{rank} ({r['score']:.3f}) Image[{r['index']}] -> "
            f"{IMAGE_LABELS[r['index']]}"
        )
    display_text_to_images(refined_cap, retrieval_results)

    print("\nPipeline complete.")
    return {
        "top_k_captions": cap_results,
        "base_caption": base_cap,
        "refined_caption": refined_cap,
        "classification": clf,
        "vqa": qa_pairs,
        "retrieval": retrieval_results,
    }


pipeline_output = full_pipeline(
    image_url=IMAGE_URLS[3],   # red sports car
    caption_pool=CAPTIONS,
    image_pool=images,
    image_pool_embs=image_embeddings,
    class_vocab=CLASS_VOCAB,
    questions=[
        "What object is this?",
        "What colour is it?",
        "Is the vehicle moving or parked?",
        "Describe the surroundings.",
    ],
    top_k=3,
)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SYSTEM SUMMARY")
print("=" * 70)
print(f"""
  Models
  ------
  Vision-Language Alignment  : openai/clip-vit-base-patch32
  Caption Generation         : Salesforce/blip-image-captioning-base
  Visual Question Answering  : Salesforce/blip-vqa-base
  Device                     : {DEVICE.upper()}

  Tasks Completed
  ---------------
  Task 1 - Image -> Top-k Caption Matching   (CLIP cosine similarity)
  Task 2 - Text  -> Image Retrieval           (CLIP cosine similarity)
  Task 3 - Zero-Shot Classification           (CLIP prompt engineering)
  Task 4 - Visual Question Answering          (BLIP-VQA)
  Task 5 - Caption Generation & Refinement    (BLIP Captioning)

  Files Saved (in outputs/)
  -------------------------
  similarity_matrix.png
  image_to_captions_*.png
  text_retrieval_*.png
  zeroshot_*.png
  vqa_results.png
  caption_refinement.png
  pipeline_query.png
  run_summary.json
""")

# ─────────────────────────────────────────────
# Save structured run summary
# ─────────────────────────────────────────────
summary = {
    "device": DEVICE,
    "image_labels": IMAGE_LABELS,
    "image_urls": IMAGE_URLS,
    "captions": CAPTIONS,
    "similarity_matrix": sim_matrix.cpu().tolist(),
    "zero_shot": [
        {
            "image": IMAGE_LABELS[idx],
            **{
                k: v
                for k, v in zero_shot_classify(images[idx], CLASS_VOCAB).items()
            },
        }
        for idx in range(len(images))
    ],
    "vqa": [
        {"image": IMAGE_LABELS[i], "question": q, "answer": a}
        for i, q, a in vqa_records
    ],
    "captions_generated": [
        {"image": IMAGE_LABELS[i], "base": b, "refined": r}
        for i, b, r in caption_results
    ],
    "pipeline": {
        "base_caption": pipeline_output["base_caption"],
        "refined_caption": pipeline_output["refined_caption"],
        "classification": {
            "predicted": pipeline_output["classification"]["predicted"],
            "confidence": pipeline_output["classification"]["confidence"],
            "all": pipeline_output["classification"]["all"],
        },
        "vqa": [
            {"question": q, "answer": a}
            for _, q, a in pipeline_output["vqa"]
        ],
        "retrieval": [
            {"index": r["index"],
             "label": IMAGE_LABELS[r["index"]],
             "score": r["score"]}
            for r in pipeline_output["retrieval"]
        ],
    },
}

with open(out_path("run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved -> {out_path('run_summary.json')}")
