# Graph-Based Pedestrian Navigation Uncertainty Estimation for Identifying Confusing Public Spaces
### Complete Research & Implementation Roadmap

---

## Part 1 — Literature Review

### 1.1 Existing Work by Category

**Pedestrian trajectory prediction.** The dominant paradigm forecasts future (x, y) positions given observed history. Social-LSTM introduced social pooling to model neighbor influence. Social-GAN added adversarial training for multimodal futures. Trajectron++ used graph-structured recurrent models conditioned on agent type and dynamics. Social-STGCNN and STAR replaced RNNs with spatio-temporal graph convolutions and transformers respectively, treating each pedestrian as a graph node updated over sliding windows. More recent work (AgentFormer, MID, EqMotion) uses diffusion or transformer backbones. The universal objective is minimizing displacement error (ADE/FDE) between predicted and ground-truth future paths.

**Crowd anomaly detection.** This line (e.g., optical-flow-based social force models, ST-AutoEncoders, deep one-class SVMs on UCSD Ped1/Ped2, Avenue, ShanghaiTech) flags frames or trajectories that deviate from a learned "normal" distribution — running, fighting, loitering, wrong-direction movement. It is framed as outlier detection: anomalies are rare, discrete events, usually violent or rule-breaking, and are evaluated with frame-level AUC on datasets curated specifically to contain rare abnormal events.

**Trajectory understanding / motif and intent modeling.** Work on intention prediction, destination prediction, and route-choice modeling (e.g., inverse reinforcement learning approaches, goal-conditioned prediction) tries to infer *where* a pedestrian is headed, not *how confidently* they are getting there.

**Crowd behavior analysis / density and flow.** Crowd counting (MCNN, CSRNet), flow prediction, and social-group detection (e.g., coherent filtering) describe aggregate crowd states — density maps, flow fields, group formation — but operate at a population level and do not attribute a per-individual cognitive/behavioral state.

**Explainable crowd analytics.** This is the least developed subfield. A handful of papers apply attention visualization or SHAP to trajectory prediction models post hoc, but explainability is rarely the design objective; it is typically a diagnostic add-on for accuracy debugging, not a tool for explaining *environmental* properties.

### 1.2 Limitations and Research Gaps

- Trajectory prediction optimizes ADE/FDE — it rewards models for predicting whatever a pedestrian does, including confused wandering, without ever measuring or exposing how confused that wandering was. A model can achieve excellent ADE/FDE on a confused trajectory if it also learns to imitate the confusion.
- Anomaly detection assumes abnormality is rare and semantically dramatic (violence, running, trespassing). Navigational confusion is common, mundane, and *not* an outlier in the statistical sense — most pedestrians in a badly-designed transit hub will hesitate, so "hesitation" is the majority behavior, not an anomaly.
- Anomaly detection is typically unsupervised on a closed world of "normal" training footage; it has no mechanism to produce a continuous, comparable score per agent, and no mechanism to explain *why* a location is difficult — only that a clip looked unusual.
- Density/flow analysis describes crowd-level phenomena (congestion, bottlenecks) but says nothing about individual wayfinding failure — a corridor can have high, smooth flow while still being confusing (people flow through it while backtracking and rerouting internally).
- No existing public dataset contains ground-truth "confusion" or "navigation uncertainty" annotations, because the construct is subjective and expensive to label at scale — this is precisely why the field has avoided it and instead substituted adjacent, more label-friendly problems (prediction accuracy, anomaly frequency).

### 1.3 Why Navigation Uncertainty Is Distinct

| Dimension | Anomaly Detection | Trajectory Prediction | Navigation Uncertainty (this project) |
|---|---|---|---|
| Frequency assumption | Rare | N/A | Common, environment-dependent |
| Output | Binary/frame flag | Future coordinates | Continuous per-agent, per-region score |
| Semantics | Rule-violating/dangerous | Motion continuation | Wayfinding difficulty |
| Granularity | Frame/clip | Trajectory | Trajectory segment + spatial zone |
| Explainability | Rare focus | Rare focus | Core deliverable |
| Ground truth | Curated abnormal clips | Observed future path | Weakly supervised proxy (this project designs it) |

### 1.4 Why This Problem Is Worth Solving

Public-space designers (airports, transit hubs, hospitals, campuses, malls) currently rely on post-occupancy surveys, expert walkthroughs, or expensive eye-tracking studies to identify confusing wayfinding zones. A trajectory-derived, camera-based Navigation Uncertainty Score would give a scalable, passive, continuously-updating diagnostic — usable during design review (simulation), construction (post-occupancy evaluation), and operations (signage placement, staffing).

### 1.5 Novelty Statement

> This project introduces **Navigation Uncertainty Estimation (NUE)** as a new task distinct from trajectory prediction and anomaly detection: a graph neural network learns to regress a continuous, per-pedestrian **Navigation Uncertainty Score (NUS)** from weakly-supervised behavioral proxy labels derived from measurable trajectory kinematics (stopping, backtracking, path inefficiency, hesitation, oscillation), aggregates these into a spatial heatmap of environment-level wayfinding difficulty, and produces feature-attributable explanations (via SHAP / GNNExplainer) for *why* a region is confusing — a capability absent from existing trajectory-prediction and crowd-anomaly literature.

---

## Part 2 — Problem Definition

### 2.1 Input

For a scene observed over a time horizon, the input is a set of pedestrian trajectories:

T = { τ₁, τ₂, ..., τ_N }, where each τᵢ = [(x₁,y₁,t₁), (x₂,y₂,t₂), ..., (x_Tᵢ, y_Tᵢ, t_Tᵢ)]

extracted either from a benchmark dataset (top-down or bird's-eye pixel/world coordinates) or from video via detection + tracking (YOLO + ByteTrack/OC-SORT → homography → world coordinates). Optionally, a static environment map/segmentation (walkable area, obstacles, POIs, entrances/exits) is available as auxiliary input.

### 2.2 Output

1. **NUS(τᵢ, t)** ∈ [0, 1]: a per-pedestrian, per-time-window Navigation Uncertainty Score.
2. **H(x, y)**: a spatial heatmap over the scene, aggregating NUS across all pedestrians who passed through each grid cell.
3. **E(τᵢ, t)**: an explanation vector (top contributing features, e.g., "high turning-angle variance + 2 stops within 3 m of intersection") for each high-NUS instance/region.

### 2.3 Assumptions

- Trajectories are reasonably complete (tracking gaps < some threshold, else the segment is discarded or interpolated).
- The scene has a fixed reference frame (homography-calibrated world coordinates), so spatial statistics are comparable across pedestrians.
- Confusion, while a private cognitive state, has **observable, measurable behavioral correlates** in motion kinematics — the central, literature-supported hypothesis (stopping, re-orientation, backtracking are established wayfinding-failure indicators in environmental psychology, e.g., Lynch's "legibility," Passini's wayfinding research, O'Neill's spatial cognition studies).
- Weak/proxy labels are noisy but directionally correct — the model learns a *relative*, not absolute, notion of uncertainty, which is sufficient for identifying and ranking problem zones.

### 2.4 Notation

- τᵢ: trajectory of pedestrian i, sampled at Δt.
- w: sliding window (e.g., 5 s) over which local features are computed.
- fᵢ,w ∈ ℝᵈ: hand-engineered feature vector for pedestrian i in window w.
- Gᵢ,w = (V, E, A): interaction graph at window w (nodes = pedestrians present, edges = spatial/social proximity, A = adjacency/edge-weight matrix).
- ŷᵢ,w = NUS: model's predicted uncertainty score.
- yᵢ,w: weak-supervision pseudo-label (Part 4).

### 2.5 Objective Function

Primary training objective — regression with a robust loss (Huber/Smooth-L1 to reduce sensitivity to noisy pseudo-labels):

L = (1/M) Σ Lδ(ŷᵢ,w − yᵢ,w) + λ₁·L_rank + λ₂·L_reg

- **L_rank**: a pairwise ranking loss (margin/RankNet-style) that enforces relative ordering between high- and low-confidence windows even when absolute pseudo-label magnitude is uncertain — this is important because the pseudo-labels are more trustworthy in *relative* ordering than absolute value.
- **L_reg**: standard weight-decay/L2 regularization.

### 2.6 Regression vs. Classification

Binary "confused / not confused" classification discards information and forces an arbitrary threshold on an inherently continuous phenomenon (a pedestrian who pauses once briefly at an intersection is not equivalent to one who loops back three times). Regression to a continuous NUS:

- preserves ranking information needed for heatmap intensity and prioritization,
- avoids brittle threshold-tuning and label-noise sensitivity at the decision boundary,
- allows downstream consumers (urban planners) to set their own operational threshold (e.g., "flag top 10% of zones") without retraining.

A classification head can still be attached as an auxiliary task (multi-task learning) to stabilize training, but the primary output must remain the continuous score.

---

## Part 3 — Dataset Analysis

| Dataset | Contents | Annotations | Trajectory format | Suitability | Limitations |
|---|---|---|---|---|---|
| **Stanford Drone Dataset (SDD)** | 60+ aerial drone videos over 8 campus scenes (intersections, plazas) | Bounding boxes + track IDs for pedestrians, cyclists, cars, etc. | Pixel-space, high density, real campus intersections | **High** — realistic pedestrian mixing at confusing intersections, large scale, top-down view ideal for graph construction | No homography/world coords provided by default (must self-calibrate); labels are object detections, not wayfinding-specific |
| **ETH** | 2 scenes (ETH, Hotel), classic pedestrian benchmark | World-coordinate trajectories, ~750 pedestrians | World coords, low density | Medium — good for validating kinematic feature pipelines, too small/simple alone | Small scene diversity, sparse crowds limit interaction-graph richness |
| **UCY** | 3 scenes (Zara1, Zara2, UCY students) | World-coordinate trajectories | World coords | Medium — same use as ETH, standard prediction-literature benchmark | Same limitations as ETH; commonly combined with ETH as "ETH/UCY" |
| **inD (Intersection Drone Dataset)** | Drone footage of 4 German intersections, vehicles + pedestrians + cyclists | High-precision tracks (world coords, cm-level via drone+CV pipeline) | World coords, real intersections | **High** — intersections are exactly the kind of decision-point geography where navigation uncertainty is expected | Focus is traffic-safety oriented (vehicle-heavy); licensing requires academic request; limited pure-pedestrian precincts (mostly street-crossing) |
| **JRDB** | Robot-egocentric multi-sensor (2D/3D lidar + stereo) campus dataset, social navigation focus | 3D pedestrian tracks + rich social context, indoor/outdoor mixed | 3D world coords, robot-centric | Medium-High — dense indoor/outdoor pedestrian mixing (halls, plazas) matches confusing-space use case; explicit social-interaction annotations useful for graph edges | Egocentric (robot-following) capture means partial/moving frame of reference, more preprocessing to get a static top-down reference; smaller pedestrian count than SDD |

### 3.1 Recommendation

Use **SDD** as the primary dataset (scale, real campus intersections/plazas with natural wayfinding decision points, high pedestrian density for rich interaction graphs) supplemented by **inD** for intersection-heavy validation and **ETH/UCY** as a small, well-understood sanity-check/prior-literature comparison set. JRDB is a strong secondary candidate for a follow-on indoor-focused extension (given effort/time constraints for a final-year project, treat it as future work rather than a Phase-1 requirement) — mention it explicitly in the proposal as planned generalization work.

### 3.2 Download / Preprocessing / Organization

```
1. Download SDD from the official Stanford CVGL page (or the Kaggle mirror) —
   video + annotations.txt per scene (bounding box, frame, track ID, "lost"/"occluded" flags).
2. Download inD from the RWTH levelXdata portal after academic registration (CSV per recording:
   tracks.csv, tracksMeta.csv, recordingMeta.csv).
3. Download ETH/UCY from the standard benchmark split repositories
   (widely mirrored, e.g., via the Social-GAN / Trajectron++ GitHub repos' data folders).

Preprocessing pipeline (scripts/preprocess_<dataset>.py):
  a. Parse raw annotation into a unified schema:
     [dataset, scene_id, agent_id, frame, timestamp, x_world, y_world, agent_type]
  b. For SDD: estimate/apply per-scene homography (planar campus scenes) to convert
     pixel coordinates to approximate metric world coordinates.
  c. Resample all tracks to a fixed Δt (e.g., 0.4s, matching common ETH/UCY convention)
     via linear interpolation; discard tracks with gaps > 1s.
  d. Filter to pedestrian class only (drop cyclists/vehicles) except where used as
     environmental context (e.g., inD vehicle tracks as obstacles).
  e. Segment each track into overlapping sliding windows (Part 4/5).
  f. Store as Parquet/HDF5 partitioned by dataset/scene for efficient loading.
```

### 3.3 Combining Multiple Datasets

Because coordinate systems, frame rates, and scene scales differ, combination requires: (a) unifying to a common Δt via resampling, (b) normalizing each scene's coordinates independently (z-score or min-max per scene, never globally, since absolute scale isn't comparable across scenes), and (c) tagging every sample with a `scene_id` so scene-relative graph construction (Part 6) and evaluation splits (train/val/test **by scene**, never by random trajectory shuffling, to test generalization) are possible.

---

## Part 4 — Weak-Supervision Label Generation

No dataset provides ground-truth confusion labels, so the Navigation Uncertainty Score pseudo-label is constructed from a weighted combination of measurable, literature-grounded proxy features computed per sliding window w for pedestrian i.

### 4.1 Component Features and Formulas

Let positions in window w be p₁...p_k with timestamps t₁...t_k.

1. **Stopping frequency (SF):** fraction of frames with speed below threshold vₘᵢₙ:
   SF = (1/k) Σ 𝟙[vⱼ < vₘᵢₙ]

2. **Speed variance (SV):** normalized variance of instantaneous speed:
   SV = Var(v₁...v_k) / (mean(v)² + ε)

3. **Path efficiency (PE):** ratio of straight-line displacement to actual path length (inverted so higher = more uncertain):
   PE_raw = ‖p_k − p₁‖ / Σ‖pⱼ₊₁ − pⱼ‖,  PE = 1 − PE_raw

4. **Turning-angle variance (TAV):** variance of heading-change angles θⱼ = angle(pⱼ₊₁−pⱼ, pⱼ−pⱼ₋₁):
   TAV = Var(θ₁...θ_{k-2})

5. **Local hesitation (LH):** count of local speed minima (deceleration-then-reacceleration events) within the window, normalized by window duration.

6. **Loop/backtrack detection (LB):** binary/continuous measure of self-intersection — using a spatial hashing/segment-intersection test on the window's path polyline, or cosine similarity between the outbound and a later sub-path's direction vectors falling below a negative threshold (near-reversal).

7. **Interaction-change rate (ICR):** rate of change in the pedestrian's k-nearest-neighbor set membership across the window (frequent following/leaving of different neighbors correlates with re-orientation behavior).

8. **Local crowd density (CD):** number of neighbors within radius r, normalized — included both as a raw feature and as a **confound control** (see 4.4).

9. **Waiting time (WT):** longest continuous low-speed run within the window, normalized by window length.

**Additional named sub-signals worth computing explicitly (fold into TAV/LB above, or track separately for interpretability):**
- **U-turn detection:** flag any turning angle > 150° as a discrete strong-reversal event — a classic "wrong way" indicator, cheap to compute and useful as a standalone explanation feature later (Part 9).
- **Oscillation:** count zero-crossings of the heading-angle derivative within the window — captures back-and-forth movement in place, distinct from a single large backtrack (LB) or generally high turning variance (TAV).

### 4.2 Normalization

Each raw component is normalized per-scene (not globally) via robust scaling: xₙₒᵣₘ = clip((x − median) / IQR, 0, 1), which is less sensitive to outlier trajectories (e.g., a person literally standing still to wait for a friend) than min-max scaling.

### 4.3 Weighting Strategy

NUS_weak = Σⱼ wⱼ · componentⱼ,ₙₒᵣₘ, Σwⱼ = 1

Two justifiable weighting approaches, both to be reported (for ablation, Part 10):

- **Equal/expert-prior weighting:** weights set from environmental-psychology literature emphasis (stopping and backtracking weighted highest, e.g., w_SF=w_LB=0.2, others 0.1–0.15) — transparent, reproducible, no circularity.
- **PCA-derived weighting:** first principal component loadings across the 9 normalized features, interpreted as the dominant shared "uncertainty axis" — data-driven, but must be validated (loadings should be positive and roughly literature-consistent; if PC1 doesn't align with prior expectation, do not use it blindly).
- **Entropy weighting (third option, worth including in the ablation):** weight each feature by how much it discriminates across the dataset — higher variance/entropy features get more weight. Data-driven and requires no manual tuning or PCA, and is well-established in multi-criteria decision-making literature. Caveat: it assumes high-variance features are meaningful signal rather than noise, so sanity-check the resulting weights the same way as the PCA loadings. A reasonable two-tier design: use entropy weighting to generate the initial automatic label (no manual labeling needed), then optionally refine weights via regression against the small human-rated validation subset (Section 4.5) if time allows.

### 4.4 Confound Control

Crowd density alone can *cause* slower speed and more stopping without confusion (e.g., queueing). The label pipeline must therefore **residualize** kinematic features against local density where theoretically justified — e.g., regress speed/stopping features on density and use residuals — OR explicitly report density as a separate covariate in later evaluation so reviewers can see the model isn't just learning "crowded = confused."

### 4.5 Alternative Label Generation Methods (discussed, not primary)

- **Self-supervised contrastive framing:** treat "efficient, direct" trajectories to a common destination as positive anchors and their pairwise deviation as an implicit uncertainty proxy — avoids hand weighting but is harder to interpret/explain (conflicts with Part 9's explainability goal).
- **Small-scale human annotation calibration:** have 3–5 human raters watch a held-out sample of trajectory replays and rate "apparent confusion" 1–5; use this as a *validation* set (not training label) to check that NUS_weak correlates with human judgment (Spearman ρ) — recommended as a lightweight validity check, not full-scale labeling.

### 4.6 Reproducibility

All formulas are closed-form functions of (x, y, t) alone (no learned components), computed identically from a documented script with fixed thresholds/window sizes reported in an appendix — any researcher re-running the pipeline on the same preprocessed tracks obtains bit-identical pseudo-labels, satisfying reproducibility.

### 4.7 Framing Statement (repeat in the thesis/paper and at defense)

> The Navigation Uncertainty Index is a quantitative summary of *trajectory kinematic irregularity*. It is not a measurement of cognitive state, emotion, or psychological confusion, and no claim of mind-reading is made or implied. High scores indicate movement patterns statistically associated with wayfinding difficulty, not a diagnosis of the individual.

This framing is what makes the weak-supervision label design defensible and publishable — every score in the project is a *movement irregularity index*, never a psychological claim. If a panel member pushes on ethics at viva, the load-bearing line to repeat is: *"We measure kinematic irregularity in movement, not psychological state — the model has no access to, and makes no claims about, what a pedestrian was thinking."*

---

## Part 5 — Feature Engineering

| Category | Features | Why useful | Approx. dim |
|---|---|---|---|
| **Motion** | speed, acceleration, jerk, heading, angular velocity, deceleration events | Direct kinematic signature of hesitation/reorientation | 6–8 |
| **Spatial** | distance to nearest wall/obstacle, distance to nearest POI/exit/signage (if map available), position relative to intersection/decision-point, local path curvature | Confusion is spatially situated — decision points are where uncertainty concentrates | 5–7 |
| **Temporal** | time since window start, time-of-day, dwell duration in current cell, window-level duration | Confusion may correlate with peak/off-peak differences, and dwell time itself is diagnostic | 3–4 |
| **Interaction (social)** | k-NN distances, relative velocity to neighbors, group-membership flag, following/leading indicator, local density | Confused pedestrians often exhibit atypical interaction — either isolating or erratically joining/leaving groups | 6–8 |
| **Environmental** | segmented walkable-area width, presence of visual obstruction (if semantic map available), signage density (if annotated) | Grounds the score in *why* a place is confusing, not just *that* a person behaved oddly | 3–5 (optional, dataset-dependent) |
| **Graph-derived** | node centrality within the interaction graph, edge-weight statistics, graph-Laplacian positional encoding | Captures how a pedestrian's behavior compares to the *local social context*, not just isolated kinematics | 4–6 |

Total raw feature dimensionality: roughly **25–35** before graph embedding, reduced via:

- **Correlation pruning** (drop one of any pair with |r| > 0.9, e.g., speed variance vs. jerk variance overlap),
- **Permutation importance** using a fast baseline (Random Forest regressor on NUS_weak) to rank features before finalizing the GNN input set,
- **PCA only for exploratory analysis**, not as the actual GNN input (keeps individual features interpretable for Part 9 explainability — this is an explicit design choice made to preserve SHAP-level interpretability, trading off some possible redundancy compression).

---

## Part 6 — Graph Construction

### 6.1 Nodes and Edges

- **Nodes V**: each pedestrian present in window w is a node; node feature vector = the pruned feature set from Part 5.
- **Edges E**: connect pedestrian pairs that are spatially/socially proximate within the window.
- **Edge weights**: inverse-distance or Gaussian kernel of inter-pedestrian distance, e.g., wᵢⱼ = exp(−d²ᵢⱼ / 2σ²), optionally modulated by relative heading alignment (pedestrians moving similarly are more socially linked than two people who happen to be near but moving oppositely).

### 6.2 Temporal Graph Construction

Two viable designs, both to be prototyped:

1. **Snapshot sequence**: one static graph Gw per sliding window, fed sequentially into a temporal aggregator (GRU/temporal-attention) over graph embeddings — simpler, matches ST-GCN-style literature.
2. **Fully dynamic graph**: nodes/edges updated at every frame with a continuous-time GNN (e.g., Temporal Graph Networks) — richer but heavier to train and harder to explain (Part 9 becomes more complex on continuous-time models). **Recommended default: snapshot sequence** for the final-year-project time budget, with dynamic-graph as a stated future-work extension.

### 6.3 Neighbor Selection: k-NN vs. Radius Graph

| | k-NN | Radius graph |
|---|---|---|
| Behavior in dense crowds | Fixed edge count avoids over-connected hubs | Can blow up edge count near dense clusters |
| Behavior in sparse areas | Forces connections to far-away, socially-irrelevant agents | Correctly yields isolated/no-edge nodes when truly alone (which is itself meaningful — isolation may correlate with either confidence or confusion depending on context) |
| Recommendation | **Use radius graph** (r ≈ 3–5 m, tunable per scene density) as primary, with a **capped k-NN fallback** (max degree cap, e.g., 8) to prevent pathological edge blow-up in very dense SDD plazas — a hybrid that is standard practice in crowd-GNN literature | |

Distance threshold r should be chosen empirically per scene (e.g., 90th-percentile nearest-neighbor distance) and reported explicitly for reproducibility.

---

## Part 7 — Model Design

### 7.1 Architecture Comparison

| Model | Captures spatial structure | Captures temporal dynamics | Attention/interpretability | Fit for this task |
|---|---|---|---|---|
| LSTM (no graph) | No | Yes | Low | Baseline only — ignores social structure entirely |
| Transformer (sequence-only) | Weak (via self-attn over time) | Yes | Medium (attention weights) | Baseline — no explicit spatial graph inductive bias |
| GCN | Yes (spectral/local) | No (needs external temporal wrapper) | Low | Basic graph baseline |
| GraphSAGE | Yes (inductive sampling) | No | Low-Medium | Good for scalability to unseen large crowds |
| GAT | Yes, with learned edge attention | No | **High** (attention weights are directly interpretable per-neighbor) | Strong candidate — attention gives free interpretability |
| ST-GCN | Yes | Yes (built-in temporal conv) | Low-Medium | Strong candidate, proven in skeleton/trajectory literature |
| Temporal Graph Networks (TGN) | Yes | Yes (continuous-time) | Medium | Powerful but heavy, harder to explain, longer training |
| Graph Transformer | Yes (global attention over graph) | Yes if combined with temporal encoding | **High** | Most expressive, most compute-hungry |

### 7.2 Recommendation

**Primary architecture: a GAT-based spatio-temporal model** — spatial GAT layers per window snapshot (to get interpretable per-neighbor attention) followed by a GRU/temporal-attention aggregator across windows. This is recommended because it (a) is well-supported in literature (feasible for a final-year timeline), (b) yields attention weights usable directly for Part 9 explainability without extra machinery, and (c) is less compute-hungry than full graph transformers or TGN. ST-GCN is proposed as a strong secondary/ablation architecture for comparison, since it is a well-established baseline in trajectory literature.

### 7.3 Network Design

```
Input:  node features Xw ∈ R^(N_w × d)   [per-window, N_w = #pedestrians present]
        adjacency/edge-weights Aw ∈ R^(N_w × N_w)  [radius graph, Part 6]

Spatial encoder (per window):
  GAT layer 1: d -> 64  (4 attention heads, concat)      + ELU + dropout(0.2)
  GAT layer 2: 64*4 -> 64 (1 head, average)               + ELU
  -> per-node spatial embedding zᵢ,w ∈ R^64

Temporal aggregator (across L windows per pedestrian, L≈6, i.e. 30s history at 5s windows):
  GRU (input 64, hidden 128) OR temporal self-attention over {zᵢ,w-L...zᵢ,w}
  -> hᵢ,w ∈ R^128

Output head:
  MLP: 128 -> 64 -> 1  (sigmoid) -> ŷᵢ,w = NUS ∈ [0,1]
  Auxiliary classification head: 128 -> 2 (low/high uncertainty, multi-task regularizer)

Loss: Huber(ŷ, y_weak) + λ1 * pairwise-rank loss + λ2 * L2 + λ3 * aux-CE
Optimizer: AdamW, lr=3e-4, cosine decay with 5-epoch warmup
Batch construction: batch by scene + time-window (graph-mini-batching via PyG's Batch),
                    batch size = 32 windows (variable node count per graph, handled by PyG)
Validation: scene-held-out split (train on N-1 scenes, validate/test on held-out scene(s))
            to measure spatial generalization, not just temporal generalization
Epochs: ~60-100 with early stopping on validation Huber loss (patience=10)
```

---

## Part 8 — Spatial Heatmap Generation

1. **Grid generation**: overlay a uniform grid (cell size ≈ 0.5–1 m, tuned to scene scale) over the calibrated world-coordinate scene.
2. **Spatial aggregation**: for every trajectory point falling in a cell, accumulate its window-level NUS (weighted by dwell time in that cell) into that cell's running statistics (mean, count, variance).
3. **Heatmap smoothing**: Gaussian kernel density smoothing (σ ≈ 1–2 cells) to avoid sparse-cell noise, and normalize by pedestrian-count-in-cell so that low-traffic cells with a single confused pedestrian aren't visually equated with well-sampled genuinely-confusing zones (report both a smoothed mean-NUS heatmap and a sample-count heatmap side by side, so viewers can judge confidence).
4. **Time aggregation**: allow filtering/aggregation by time-of-day or day-of-week if timestamps support it, to detect whether confusion is context-dependent (e.g., only during rush hour crowding).
5. **Interactive visualization**: a web dashboard (Part 11/13) overlaying the heatmap on the scene's reference image/map, with click-to-inspect showing the top contributing trajectories and their explanation (Part 9) for that cell. Use a **sequential colormap** (low = blue/green, high = red) rather than a diverging red/green scale, to avoid visually implying a "good/bad" binary on what is a continuous score.
6. **Confusing-zone detection**: threshold the smoothed heatmap at a statistically justified cutoff (e.g., cells above the 90th percentile of mean-NUS **and** with sample count above a minimum-confidence threshold) to output a discrete list of flagged zones with bounding polygons for reporting purposes.

---

## Part 9 — Explainable AI

| Method | What it explains | Fit here |
|---|---|---|
| **SHAP** (KernelSHAP/TreeSHAP on a distilled surrogate, or GradientSHAP on the GNN) | Per-instance feature attribution (which of the 25–35 engineered features drove this NUS) | **Primary** — directly answers "why was this pedestrian scored uncertain," matches the project's stated deliverable |
| **Integrated Gradients** | Attribution along the path from a baseline to the actual input, differentiable-model-native | Secondary — useful cross-check against SHAP for the neural components |
| **Attention visualization** | Which neighbors/time-windows the GAT/temporal-attention weighted most | **Free interpretability** from the chosen architecture — visualize attention as edge thickness overlaid on the trajectory replay |
| **Graph explainability (GNNExplainer)** | Which subgraph (which neighbors + which edges) is most responsible for a node's prediction | Useful for explaining *social* causes of uncertainty (e.g., "confusion driven by a large group blocking sightline") distinct from purely kinematic causes |

### 9.1 Per-Pedestrian Explanation

For each flagged high-NUS instance, generate a structured explanation: top-3 SHAP features (e.g., "turning-angle variance," "loop/backtrack score," "stopping frequency") + top-attended neighbor(s)/edge(s) from GAT + a short natural-language template summarizing them (e.g., "Pedestrian showed 3 direction reversals near a group of 4, with 40% of the window spent below walking speed").

### 9.2 Environment-Level Explanation

Aggregate SHAP attributions across all instances within a flagged heatmap cell to produce a **cell-level dominant-cause profile** (e.g., "This intersection's high NUS is driven primarily by turning-angle variance and loop detection, not by crowd density" — directly useful to a wayfinding designer, since it distinguishes "confusing signage/layout" from "just crowded").

---

## Part 10 — Evaluation

- **Regression metrics**: MAE, RMSE, R² between predicted NUS and (a) held-out weak labels, (b) the small human-rated validation subset (Part 4.5).
- **Ranking metrics**: Spearman/Kendall-τ correlation between predicted and weak-label rankings; NDCG if treating "most confusing trajectories" as a ranked retrieval task.
- **Spatial evaluation**: correlation between the model's heatmap and an independent proxy of known difficult zones (e.g., self-reported "got lost" survey data if collectable, or expert wayfinding-audit annotations of the same physical intersections if available in facilities literature) — explicitly flagged as optional/aspirational depending on data access.
- **Ablation studies**: remove each feature category (Part 5) one at a time; remove graph structure entirely (fall back to per-agent LSTM) to quantify the graph's contribution; compare radius-graph vs. k-NN; compare weak-label weighting schemes (Part 4.3).
- **Baseline comparisons**: (1) heuristic baseline = raw weighted-feature-sum without any learned model, (2) LSTM-only, (3) GCN, (4) ST-GCN, (5) proposed GAT+temporal model.
- **Cross-dataset generalization**: train on SDD, test zero-shot on inD (and vice versa) to demonstrate the model isn't scene-overfit — a key novelty-strengthening experiment for publication.
- **Statistical significance**: paired bootstrap resampling (1000 resamples) over test-set windows to report confidence intervals on MAE/Spearman-ρ differences between the proposed model and each baseline; report p-values for the primary comparison.

---

## Part 11 — System Architecture

```
 ┌─────────────┐    ┌────────────────┐    ┌────────────────────┐
 │  Raw Dataset │ →  │ Preprocessing   │ →  │ Feature Extraction  │
 │ (SDD/inD/    │    │ (calibration,   │    │ (motion, spatial,   │
 │  ETH/UCY)    │    │  resampling,    │    │  temporal, social,  │
 │              │    │  windowing)     │    │  environmental)     │
 └─────────────┘    └────────────────┘    └──────────┬──────────┘
                                                       ↓
 ┌────────────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ Weak Label          │ ← │ Graph             │ ← │ (features)   │
 │ Generation (NUS_weak)│  │ Construction      │   │              │
 └──────────┬──────────┘   │ (nodes/edges/wts) │   └──────────────┘
            ↓               └────────┬─────────┘
            └────────────────────────┼───────────────┐
                                      ↓                ↓
                          ┌────────────────────┐  (labels feed loss)
                          │  GNN Model          │
                          │ (GAT spatial +      │
                          │  temporal aggr.)    │
                          └──────────┬──────────┘
                                     ↓
                     ┌───────────────────────────────┐
                     │ Navigation Uncertainty         │
                     │ Prediction (per pedestrian)     │
                     └───────┬───────────────┬────────┘
                             ↓                ↓
                ┌────────────────────┐  ┌────────────────────┐
                │ Heatmap Generation  │  │ Explainability       │
                │ (grid aggregation,  │  │ (SHAP, attention,    │
                │  smoothing)         │  │  GNNExplainer)        │
                └──────────┬──────────┘  └──────────┬───────────┘
                           └──────────┬──────────────┘
                                      ↓
                          ┌────────────────────────┐
                          │ Dashboard (web,          │
                          │  interactive heatmap +   │
                          │  per-instance explain)   │
                          └────────────────────────┘
```

---

## Part 12 — Implementation

- **Language**: Python 3.10+.
- **Deep learning framework**: PyTorch + **PyTorch Geometric (PyG)** for graph batching/GAT/GCN/GraphSAGE layers (industry-standard, well-documented for exactly this graph-batch-with-variable-node-count scenario).
- **Graph/scientific libraries**: NetworkX (prototyping graph construction logic before PyG-izing), SciPy (KDTree for radius-graph neighbor queries), NumPy/Pandas.
- **Feature/eval libraries**: scikit-learn (baselines, PCA, permutation importance, RandomForest baseline), SHAP library, Captum (Integrated Gradients, native PyTorch support).
- **Tracking (if extending to live video, Part 13)**: Ultralytics YOLOv8/v9 for detection, ByteTrack or OC-SORT for association.
- **Visualization/dashboard**: Plotly/Dash or a lightweight Flask + Leaflet.js (for georeferenced heatmap overlay) — Flask+SocketIO if live/streaming updates are wanted, matching prior project experience with Flask+SocketIO dashboards.
- **Folder structure**:

```
navigation-uncertainty/
├── data/
│   ├── raw/{sdd,ind,eth_ucy}/
│   ├── processed/{scene}/tracks.parquet
│   └── weak_labels/{scene}/nus_weak.parquet
├── src/
│   ├── preprocessing/        (per-dataset parsers + calibration)
│   ├── features/              (motion.py, spatial.py, temporal.py, social.py, env.py)
│   ├── labeling/               (weak_label_pipeline.py)
│   ├── graph/                  (graph_builder.py, radius_knn_hybrid.py)
│   ├── models/                 (gat_temporal.py, baselines/{lstm.py, gcn.py, stgcn.py})
│   ├── training/                (train.py, losses.py, config.yaml)
│   ├── evaluation/               (metrics.py, ablations.py, significance.py)
│   ├── explainability/            (shap_explain.py, gnn_explainer.py, attention_viz.py)
│   └── heatmap/                    (grid_aggregate.py, smoothing.py)
├── dashboard/                        (Flask app + templates + static heatmap assets)
├── notebooks/                          (EDA, ablation result plots)
├── configs/                              (per-experiment YAML configs)
└── reports/                                (generated figures, paper draft assets)
```

- **GPU requirements**: a single mid-range GPU (e.g., RTX 3060/4060, 8–12 GB VRAM) is sufficient given per-window graph sizes are small (tens to low hundreds of nodes); no need for multi-GPU.
- **Training time estimate**: with SDD + inD + ETH/UCY combined (~tens of thousands of windows), expect roughly 2–5 hours per full training run (60–100 epochs) on a single consumer GPU; ablation sweeps (Part 10) multiply this by the number of configurations, so budget for automated overnight batch runs.

---

## Part 13 — Deployment

- **REST API**: a FastAPI service exposing `/predict` (accepts a batch of recent trajectory windows, returns NUS + explanation) and `/heatmap` (returns current cached grid heatmap for a scene).
- **Web dashboard**: the Flask/Dash app from Part 12, showing the live/static heatmap overlay, click-to-inspect per-cell explanations, and a filter panel (time-of-day, date range).
- **Live video inference vs. precomputed trajectories**: recommend a **two-mode design**. Mode A (research/evaluation mode) operates directly on dataset-provided precomputed trajectories (SDD/inD/ETH-UCY) — this is what should be used for all model development, ablations, and the paper's core results, since it removes tracking-error confounds from the uncertainty-estimation evaluation. Mode B (deployment/demo mode) adds a YOLOv8 + ByteTrack/OC-SORT front-end that converts live camera video into world-coordinate tracks (via a pre-calibrated homography for the deployed camera) feeding the same downstream pipeline — this demonstrates real-world applicability for the demo/viva but should be clearly reported as a secondary, lower-precision pathway (tracking noise will inflate apparent "uncertainty" via ID switches/jitter, which must be explicitly discussed as a limitation, not hidden).
- **Heatmap visualization in deployment**: cache and incrementally update grid statistics rather than recomputing from scratch on every request, for real-time dashboard responsiveness.

---

## Part 14 — Novel Contributions

**Already solved by existing research:** trajectory prediction accuracy (ADE/FDE), crowd anomaly detection for rare/dangerous events, crowd density/flow estimation, group detection, basic GNN architectures for pedestrian modeling (GAT/ST-GCN/social pooling).

**This project's contributions:**
- *Research contribution*: formalizing Navigation Uncertainty Estimation as a distinct, continuous-valued task with an explicit, reproducible weak-supervision label design (Part 4) — this is the paper's central novel claim.
- *Research contribution*: environment-level, feature-attributable explanation of *why* a zone is confusing (dominant-cause profiling, Part 9.2), which existing crowd-analytics explainability work does not target.
- *Research contribution*: cross-dataset generalization evaluation for a wayfinding-difficulty score (not previously benchmarked, since the task itself is new).
- *Engineering contribution*: the full pipeline (weak labeling → radius/kNN hybrid graph construction → GAT-temporal model → heatmap → SHAP/GNNExplainer dashboard) as a reusable open-source toolkit.
- *Engineering contribution*: dataset harmonization scripts unifying SDD/inD/ETH-UCY into one schema — useful independent of this specific task.

**Publication suitability assessment:** the novelty (new task formulation + weak-supervision design + cross-dataset validation + explainability) is appropriate for an **IEEE-sponsored conference workshop or a Scopus-indexed conference** (e.g., ITSC-adjacent workshops, CVPR/ICCV workshops on crowd/pedestrian behavior, or a mid-tier Springer LNCS conference) rather than a top-tier main-track venue on a first submission, given the label is weakly supervised rather than ground-truth. It is comfortably strong as a **final-year project** and a credible **first conference paper**, with a path to a stronger journal submission (Scopus-indexed) after incorporating a small-scale human-validation study (Part 4.5) to strengthen label credibility.

---

## Part 15 — Deliverables & Roadmap

### 15.1 Week-by-Week Plan (assuming ~16-week semester)

| Weeks | Milestone |
|---|---|
| 1–2 | Literature review write-up, finalize problem definition, dataset access requests (esp. inD academic license) |
| 3–4 | Dataset download, unified preprocessing pipeline, calibration/homography for SDD scenes |
| 5 | Feature engineering module (Part 5) implemented + unit-tested against a few hand-checked trajectories |
| 6 | Weak-label pipeline (Part 4) implemented; sanity-check distributions per scene |
| 7 | Graph construction module (radius + capped-kNN hybrid); visualize sample graphs |
| 8–9 | Baseline models (heuristic, LSTM, GCN) trained and evaluated |
| 10–11 | Primary GAT-temporal model implemented and trained; hyperparameter tuning |
| 12 | ST-GCN comparison model; ablation study sweep |
| 13 | Heatmap generation + smoothing + visualization |
| 14 | Explainability module (SHAP, attention viz, GNNExplainer) + per-instance/per-zone explanation templates |
| 15 | Dashboard integration (Flask/Dash), cross-dataset generalization experiment, statistical significance tests |
| 16 | Final report/paper writing, figure generation, buffer for reruns |

### 15.2 Risks & Backup Plans

| Risk | Backup Plan |
|---|---|
| inD academic license delayed/denied | Proceed with SDD + ETH/UCY only; report inD as planned future work |
| Weak-label pseudo-scores don't separate meaningfully (low variance) | Fall back to the PCA-derived weighting scheme; add the small human-rating validation set earlier to recalibrate weights |
| GAT-temporal model underperforms simpler baselines | This is itself a valid, reportable ablation finding — pivot narrative toward "graph structure vs. temporal structure" analysis rather than treating it as failure |
| SDD homography calibration proves inaccurate | Fall back to reporting results in pixel-space with explicit unit caveats, restricting cross-dataset generalization claims to ETH/UCY/inD (already metric) |
| Time runs short for full explainability module | Prioritize SHAP + attention visualization (cheapest to implement); treat GNNExplainer as a stretch goal |

### 15.3 Expected Outputs

- Open-source repository (structure in Part 12) with reproducible preprocessing, training, and evaluation scripts.
- Trained model checkpoints for SDD (+ inD/ETH-UCY where feasible) with reported metrics (Part 10).
- Interactive dashboard demo (video walkthrough for viva if live deployment isn't feasible on defense day).
- Draft paper (IEEE/Springer conference format) covering Parts 1–10 content, suitable for submission to a relevant workshop/conference.

### 15.35 Limitations (Be Honest — include explicitly in the write-up)

- **Dataset limitations:** SDD scenes are campus environments — findings may not generalize to airports, hospitals, or transit stations without new data.
- **Weak supervision limitations:** the label is a proxy; a high NUS guarantees kinematic irregularity, not confusion. This must never be overstated in the write-up.
- **Behavior ambiguity:** stopping/reversing can stem from social interaction, phone use, photography, or fatigue — the model cannot distinguish cause, only correlate pattern.
- **Label quality:** since there's no independent ground truth, quantitative validation is inherently limited to proxy checks (the small manual-review subset, known-confusing-intersection sanity checks).
- **False positives:** crowded but efficient spaces may trigger irregularity features (stopping due to crowding, not confusion) — density-normalization (Part 4.4) is a needed mitigation, not a solved problem.
- **Ethics/privacy:** even with careful framing, any system inferring "confusion-like" states from surveillance-style trajectory data raises privacy/surveillance concerns that should be explicitly addressed in an ethics section. Recommendation: anonymized, aggregate-only outputs (heatmaps) in any real deployment — never expose per-individual "confusion scores."

### 15.4 Future Work

Continuous-time dynamic graphs (TGN) instead of snapshot sequences; incorporating JRDB for indoor generalization; live-camera deployment validation study with real human wayfinding surveys to replace/strengthen the weak-supervision label; multi-camera multi-view scene fusion for larger real-world spaces (airports, transit hubs) beyond single-camera-covered scenes.
