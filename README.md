# Cross-Modal Image-Text Understanding with CLIP & BLIP

A multimodal pipeline that performs **image-text alignment, retrieval, zero-shot
classification, visual question answering, and caption generation** using
OpenAI CLIP and Salesforce BLIP.

This is a Python port of the original Colab notebook
(`BT22CSD002_Cross_Modal_Image_Text_Understanding_using_CLIP_and_Flamingo.ipynb`)
re-run on a fresh dataset (mountains, beaches, robots, sports cars, planes,
basketball, food, and cityscapes).

---

## Models Used

| Task | Model |
|------|-------|
| Vision-Language alignment | `openai/clip-vit-base-patch32` |
| Caption generation | `Salesforce/blip-image-captioning-base` |
| Visual question answering | `Salesforce/blip-vqa-base` |

---

## Dataset

Eight Unsplash images covering different visual domains:

| # | Label | Reference Caption |
|---|-------|-------------------|
| 0 | mountain      | A snow-capped mountain landscape under a clear sky |
| 1 | beach sunset  | A peaceful beach during a colourful sunset |
| 2 | robot         | A humanoid robot showcasing modern technology |
| 3 | sports car    | A bright red sports car parked on a road |
| 4 | airplane      | A passenger airplane flying through the sky |
| 5 | basketball    | Players competing in an indoor basketball match |
| 6 | burger        | A delicious gourmet burger served on a plate |
| 7 | city skyline  | A city skyline glowing with lights at night |

---

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision transformers Pillow matplotlib requests
.venv/bin/python multimodal_clip_blip.py
```

All artefacts are written to `outputs/`.

---

## Pipeline Tasks

### Task 1 — CLIP Cosine Similarity Matrix

The 8 images are scored against the 8 reference captions; the diagonal should
be brightest if the model aligns the pairs correctly.

![Similarity matrix](outputs/similarity_matrix.png)

---

### Task 2 — Image → Top-k Captions

For a given image, rank all reference captions by CLIP cosine similarity.

| Mountain | Sports car | Basketball |
|---|---|---|
| ![](outputs/image_to_captions_mountain.png) | ![](outputs/image_to_captions_sports_car.png) | ![](outputs/image_to_captions_basketball.png) |

---

### Task 3 — Text → Image Retrieval

A text query is encoded with CLIP and matched against the image pool.

**Query: "snowy mountain peak"**
![](outputs/text_retrieval_snowy_mountain_peak.png)

**Query: "sunset over the ocean"**
![](outputs/text_retrieval_sunset_over_the_ocea.png)

**Query: "a fast red car"**
![](outputs/text_retrieval_a_fast_red_car.png)

**Query: "a tasty burger meal"**
![](outputs/text_retrieval_a_tasty_burger_meal.png)

**Query: "lights of a big city at night"**
![](outputs/text_retrieval_lights_of_a_big_city.png)

---

### Task 4 — Zero-Shot Classification

Each image is classified into one of
`["mountain", "beach", "robot", "car", "airplane", "basketball", "burger", "city", "forest", "river"]`
using CLIP prompt engineering — no training required.

| Mountain | Beach | Robot | Sports Car |
|---|---|---|---|
| ![](outputs/zeroshot_mountain.png) | ![](outputs/zeroshot_beach_sunset.png) | ![](outputs/zeroshot_robot.png) | ![](outputs/zeroshot_sports_car.png) |

| Airplane | Basketball | Burger | City Skyline |
|---|---|---|---|
| ![](outputs/zeroshot_airplane.png) | ![](outputs/zeroshot_basketball.png) | ![](outputs/zeroshot_burger.png) | ![](outputs/zeroshot_city_skyline.png) |

---

### Task 5 — Visual Question Answering (BLIP-VQA)

Free-form Q&A grounded in the image.

![](outputs/vqa_results.png)

---

### Task 6 — Caption Generation & Refinement (BLIP)

Base captions vs. captions conditioned on the prompt
`"a detailed photo of"`.

![](outputs/caption_refinement.png)

---

## Full Pipeline Demo (Sports Car)

End-to-end run on a single query image, chaining all five tasks:

1. Top-k caption matching (CLIP)
2. BLIP caption generation + refinement
3. Zero-shot classification (CLIP)
4. Visual Q&A (BLIP-VQA)
5. Text→image retrieval using the BLIP-generated caption

| Query image | Top-k captions | Zero-shot |
|---|---|---|
| ![](outputs/pipeline_query.png) | ![](outputs/image_to_captions_Pipeline_Query.png) | ![](outputs/zeroshot_Pipeline_Query.png) |

Retrieval using the generated caption:

![](outputs/text_retrieval_a_detailed_photo_of_.png)

---

## Output Files

All files land in `outputs/`:

| File | Description |
|---|---|
| `similarity_matrix.png` | 8×8 CLIP cosine similarity heatmap |
| `image_to_captions_*.png` | Top-k caption matches per query image |
| `text_retrieval_*.png` | Top-k images per text query |
| `zeroshot_*.png` | Zero-shot class probabilities per image |
| `vqa_results.png` | Visual Q&A panel |
| `caption_refinement.png` | Base vs. refined BLIP captions |
| `pipeline_query.png` | Query image used in the full-pipeline demo |
| `run_summary.json` | Structured numerical results (similarity matrix, classifications, VQA, captions, retrieval) |

---

## File Layout

```
genai/
├── multimodal_clip_blip.py     # main script
├── README.md                   # this file
├── outputs/                    # all generated images + JSON summary
└── BT22CSD002_Cross_Modal_Image_Text_Understanding_using_CLIP_and_Flamingo.ipynb
```
