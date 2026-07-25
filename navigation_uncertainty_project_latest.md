# Graph-Based Pedestrian Navigation Uncertainty Estimation for Identifying Confusing Public Spaces

---

## Abstract

Public spaces — transit hubs, campuses, malls, airports — are often navigated inefficiently not because people are lost in a legal sense, but because the *space itself* signals ambiguously (poor signage, symmetric layouts, occluded sightlines). No dataset labels pedestrians as "confused." What we *do* have are large public trajectory datasets (SDD, ETH/UCY, inD) capturing raw (x, y, t) motion. This project reframes "confusion" as **Navigation Uncertainty** — a behaviorally-grounded, weakly-supervised signal derived from motion irregularities (stopping, backtracking, heading oscillation, inefficient paths) rather than a psychological claim. We build weak labels from trajectory kinematics, train regression and graph-based models to predict this uncertainty score, and aggregate it spatially into a **navigation difficulty heatmap** that flags physically confusing zones in a space. The contribution is a full pipeline — from raw trajectories to explainable, spatially-actionable uncertainty maps — usable by urban planners and facility designers, without ever claiming to read minds.

**Important framing note (repeated throughout):** every score in this project is a *movement irregularity index*, not a measurement of cognitive or emotional state. This distinction is what makes the label design defensible and publishable.

---

## 1. Introduction

**Pedestrian trajectory prediction** is a mature CV/ML subfield: given past positions, predict future positions (Social-LSTM, Social-GAN, Trajectron++). These models are good at predicting *where* people go, but not at flagging *why a space made someone move erratically*.

**Navigation uncertainty**, as defined here, is the degree to which a trajectory shows signs of not knowing the intended path: excess stopping, direction reversals, wandering, inefficient routing relative to shortest path. This is observable and quantifiable directly from (x, y, t) — no extra sensors needed.

**Why trajectory behavior matters:** it's the only signal available at the scale public datasets provide. Eye-tracking or self-report data doesn't exist at scale for public spaces; trajectories do (millions of pedestrian-frames across SDD/ETH/UCY/inD).

**Why confusion labels don't exist:** labeling "confusion" requires either self-report (impossible at dataset scale/retroactively) or expert annotation (subjective, expensive, ethically fraught if framed as reading psychological state from video). This is precisely why we build the label from measurable kinematics instead — this is the core methodological move of the whole project.

---

## 2. Problem Statement

**Problem:** Given a pedestrian trajectory in a public space, estimate a continuous score representing how much the person's movement exhibits uncertainty/inefficiency behavior, and aggregate these scores spatially to identify zones in the environment that are disproportionately confusing.

**Why difficult:**
- No ground truth for "confusion" — the target itself must be engineered (weak supervision).
- Movement irregularity is multi-causal: someone might stop to check their phone, greet a friend, or take a photo — not because they're lost. The signal is *noisy by construction*.
- Class/behavior imbalance — most trajectories are efficient; irregular ones are rare.
- Generalization across datasets with different camera angles, frame rates, and pedestrian densities.

**Why still worth researching:** even a noisy, weakly-supervised uncertainty signal is useful in aggregate. Individual-level noise cancels out when heatmapped over hundreds of trajectories per zone — the spatial aggregation is what makes weak labels acceptable here.

---

## 3. Existing Datasets

| Dataset | Data Type | Annotations | Advantages | Disadvantages | Suitability |
|---|---|---|---|---|---|
| **Stanford Drone Dataset (SDD)** | Top-down drone video, 8 campus scenes | Bounding boxes, agent type, trajectories | Large scale, multiple scene types (intersections, plazas), naturalistic crowd behavior | Campus-only, some scenes not conducive to wandering/confusion behavior | **High** |
| **ETH** | Fixed camera, 2 scenes | Pedestrian trajectories, some group labels | Classic benchmark, well-studied, easy to reproduce baselines | Small scale, limited scene diversity | Medium |
| **UCY** | Fixed camera, 3 scenes | Pedestrian trajectories | Adds crowd density variety over ETH | Small, older annotation quality | Medium |
| **inD** | Drone, intersections (Germany) | Very precise trajectories (cm-level), road users | Extremely high positional accuracy | Vehicle/intersection focused, less pedestrian-space wandering behavior | Low–Medium |
| **JRDB** (optional) | Robot-mounted 360° sensors, indoor+outdoor | 3D bounding boxes, trajectories | Rich multi-modal (LiDAR + RGB), indoor navigation scenarios | Complex to preprocess, less "public space" framing | Optional/stretch |

**Recommendation: Stanford Drone Dataset (SDD).** Justification: top-down view removes perspective distortion (makes heading/turning-angle computation cleaner), scenes include plazas and intersections where wandering/hesitation naturally occurs, and its scale supports weak-label learning at the volume needed to average out per-trajectory label noise. ETH/UCY can be used as a secondary generalization test set.

---

## 4. Weak Supervision

**What it is:** training with labels that are noisy, indirect, or programmatically generated rather than hand-annotated ground truth.

**How it differs from supervised learning:** supervised learning assumes labels are the actual target of interest, collected by a reliable process (human annotation, sensor ground truth). Weak supervision assumes the label is a *proxy* — correlated with, but not identical to, the true underlying concept.

**How labels are generated here:** a rule-based/statistical function of trajectory kinematics (Section 7) produces a continuous uncertainty score per trajectory segment. This becomes the training target for downstream models.

**Why it's acceptable in research:** weak supervision is an established, well-published paradigm (Snorkel/data programming, distant supervision in NLP, noisy student training in vision). The key defensible move is being explicit that the label is a *proxy target*, evaluating on downstream utility (does the score correlate with independently-verifiable signals like stated wayfinding difficulty in small validation studies, or known confusing intersections), rather than claiming label correctness.

**Examples from other domains:**
- Distant supervision in relation extraction (Mintz et al., 2009) — labels derived from knowledge base heuristics, not human annotation.
- Snorkel/data programming (Ratner et al., 2017) — combining weak labeling functions into probabilistic labels.
- Self-supervised depth estimation — using photometric consistency as a proxy signal instead of LiDAR ground truth.

---

## 5. Proposed Methodology

| Phase | Purpose | Input | Output | Key Methods |
|---|---|---|---|---|
| 1. Preprocessing | Clean, align, and standardize raw trajectory data | Raw SDD annotations | Cleaned (x,y,t) trajectories per agent | Interpolation, coordinate normalization, frame-rate standardization |
| 2. Segmentation | Break long tracks into fixed-length windows for consistent feature computation | Cleaned trajectories | Trajectory segments (e.g. 3–5 sec windows) | Sliding window w/ overlap |
| 3. Feature extraction | Convert raw motion into interpretable behavioral signals | Segments | Feature vectors (Section 6) | Kinematic + geometric computation |
| 4. Uncertainty score generation | Build weak-supervision target | Features | Continuous uncertainty label (0–1) | Weighted combination / entropy / learned weighting (Section 7) |
| 5. Baseline ML models | Establish predictive floor: can uncertainty be predicted from raw features? | Features + label | Predicted score | Random Forest, XGBoost, MLP, Linear Regression |
| 6. Future uncertainty prediction | Predict uncertainty ahead of time from partial trajectory, not just describe it retroactively | Partial trajectory | Forecasted uncertainty | Sequence models (LSTM/Transformer) |
| 7. Graph Neural Networks | Model social/spatial interaction effects on uncertainty (crowding, blocking) | Multi-agent scene graph | Uncertainty per node, interaction-aware | ST-GCN / GAT (Section 9) |
| 8. Explainability | Make predictions interpretable for the "why" | Trained model + input | Feature attributions | SHAP, GNNExplainer |
| 9. Heatmap generation | Aggregate per-trajectory scores spatially | Scored trajectories + coordinates | 2D confusion heatmap of the scene | Grid binning + smoothing |

Each phase gates the next: features (3) can't be built without clean segments (1–2); the label (4) can't be trained on until features exist; GNNs (7) are only worth the added complexity once simple baselines (5) show the signal is learnable at all; explainability (8) only makes sense once there's a non-trivial model to explain; the heatmap (9) is the final synthesis product that all prior phases feed into.

---

## 6. Behavioral Feature Engineering

For each feature: definition → formula → interpretation.

| Feature | Formula (intuition) | Interpretation |
|---|---|---|
| **Speed** | $v_t = \frac{\|p_t - p_{t-1}\|}{\Delta t}$ | Low/variable speed can indicate hesitation |
| **Acceleration** | $a_t = \frac{v_t - v_{t-1}}{\Delta t}$ | Sudden deceleration = possible confusion moment |
| **Stop frequency** | count of frames where $v_t < \epsilon$ | More stops = more re-orientation events |
| **Heading angle** | $\theta_t = \text{atan2}(y_t - y_{t-1}, x_t - x_{t-1})$ | Direction of travel at each step |
| **Heading variance** | $\text{Var}(\theta_{t..t+w})$ | High variance = erratic direction changes |
| **Turning angle** | $\|\theta_t - \theta_{t-1}\|$ | Sharp turns per step |
| **U-turn detection** | turning angle $> 150°$ | Strong reversal signal, classic "wrong way" indicator |
| **Path efficiency** | $\frac{\text{straight-line distance}}{\text{actual path length}}$ | Low ratio = wandering / inefficient route |
| **Oscillation** | zero-crossings of heading derivative | Back-and-forth movement in place |
| **Hesitation** | stop-then-resume events within short window | Classic "pause to reconsider" pattern |
| **Local density** *(optional)* | # neighbors within radius $r$ | Controls for crowding as a confound |
| **Social interaction** *(optional)* | relative velocity/heading vs. neighbors | Distinguishes "following someone" from independent wandering |

Python implementation idea: compute all of these with `numpy`/`pandas` on windowed (x,y,t) arrays — no exotic libraries needed except for GNN-stage features later.

**Advantages:** all features are interpretable, computable from raw (x,y,t) with no extra sensors, and each has a clear physical meaning that supports the "this is not mind-reading" framing.

**Limitations:** none of these features can distinguish *why* a stop happened (confusion vs. social vs. incidental). This ambiguity is inherent and must be stated explicitly as a limitation, not hidden.

---

## 7. Navigation Uncertainty Index

This is the label-generation core of the weak supervision approach — get this right and everything downstream is sound; get it wrong and the whole pipeline inherits bias.

**Why we need it:** without a target, this is unsupervised behavior clustering, not a "confusion/uncertainty estimation" project. The index converts feature vectors into a single trainable score.

**Mathematical definition (baseline form):**

$$U = \sum_{i=1}^{n} w_i \cdot \hat{f}_i$$

where $\hat{f}_i$ are normalized (0–1) behavioral features and $w_i$ are weights, $\sum w_i = 1$.

**Normalization:** min-max or z-score followed by sigmoid squashing per feature, computed per-scene (not globally) to avoid scenes with generally faster/slower pedestrian flow biasing the score.

**Weighting approaches — compare before choosing:**

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| Manual weights | Researcher assigns weights by domain reasoning | Transparent, easy to defend in a thesis defense | Subjective, hard to justify precisely |
| **Entropy weighting** | Weight features by how much they discriminate (higher variance/entropy → more weight) | Data-driven, no manual tuning, well-established (used in multi-criteria decision-making) | Assumes high-variance features are meaningful, not noise |
| AHP (Analytic Hierarchy Process) | Pairwise comparison matrix → derived weights | Structured, used in multi-criteria decision literature | Still ultimately subjective (pairwise judgments), adds complexity for unclear benefit here |
| Fuzzy logic | Rule-based fuzzy membership functions combine features | Naturally handles ambiguity/gradation (fits "uncertainty" framing conceptually) | Rule design is itself subjective; harder to defend as objective |
| Learned weights (regression) | Fit weights against a small validated subset (e.g. manually reviewed "clearly confused" clips) | Data-driven, improvable, standard ML practice | Requires *some* validation labels — even a small hand-labeled subset |
| Bayesian combination | Treat each feature as weak evidence, combine via probabilistic model | Naturally expresses uncertainty *about* the uncertainty score (confidence intervals) | Most complex to implement and explain |

**Recommended approach:** start with **entropy weighting** for the initial automatic label (objective, no manual labeling needed) as the primary weak-supervision target. Then, if time allows, validate/refine with a **small manually-reviewed subset** (50–100 clips) to sanity-check and optionally learn better weights via regression. This gives a two-tier defensible design: unsupervised-but-principled labels, refined by minimal supervision.

**Critical framing statement (repeat in the actual thesis):**
> The Navigation Uncertainty Index is a quantitative summary of *trajectory kinematic irregularity*. It is not a measurement of cognitive state, emotion, or psychological confusion, and no claim of mind-reading is made or implied. High scores indicate movement patterns statistically associated with wayfinding difficulty, not a diagnosis of the individual.

---

## 8. Machine Learning Pipeline

**Regression vs. Classification:** Regression is preferred here because the uncertainty index is inherently continuous — forcing it into discrete classes (e.g., "confused"/"not confused") throws away information and reintroduces the exact binary framing this project is designed to avoid.

**Baseline models:**
- **Linear Regression** — establishes whether the relationship is even roughly linear; fast sanity check.
- **Random Forest** — handles non-linear feature interactions, robust to outliers, gives feature importance for free.
- **XGBoost** — typically the strongest tabular baseline; good for a leaderboard-style comparison.
- **MLP** — tests whether a simple neural net captures nonlinearities better than tree ensembles; also a stepping stone toward the sequence/graph models later.

**Evaluation metrics:**
- **MAE** — average absolute error; interpretable in the same units as the score.
- **RMSE** — penalizes large errors more; useful for flagging bad outlier predictions.
- **R²** — proportion of variance explained; overall fit quality.
- **Spearman correlation** — since the *ranking* of confusing zones matters more than exact score values for the heatmap use-case, rank correlation is arguably the most important metric here.

---

## 9. Spatio-Temporal Graph Neural Networks

**Why GNNs are useful:** individual trajectory features ignore *social context* — a pedestrian slowing down because a crowd is blocking their path looks kinematically identical to one slowing down because they're lost. Modeling pedestrians as nodes in a graph, with edges to nearby others, lets the model learn to separate social-induced slowdowns from wayfinding-induced ones.

**Graph construction:** nodes = pedestrians present in a time window; edges = proximity (within radius $r$) or visibility.

**Node features:** the kinematic features from Section 6.

**Edge features:** relative distance, relative velocity, relative heading.

**Dynamic graphs:** the graph structure changes per frame/window as people enter/leave — requires temporal graph models, not static GNNs.

**Potential architectures:**

| Model | Fit for this task |
|---|---|
| **ST-GCN** | Good for fixed-topology spatio-temporal data (originally skeleton action recognition); less natural for dynamically-changing pedestrian graphs |
| **GraphSAGE** | Inductive, handles new/unseen nodes well — good for generalizing across scenes |
| **GAT** | Attention over neighbors — naturally gives an interpretable "who influenced this uncertainty" signal |
| **TGAT / TGN** | Purpose-built for dynamic/temporal graphs — best structural fit for entering/leaving pedestrians |

**Recommendation:** **TGN (Temporal Graph Networks)** for the main model, since pedestrian graphs are inherently dynamic (nodes appear/disappear), with **GAT-style attention** as the aggregation mechanism for interpretability. If TGN proves too complex to implement in the available time, GraphSAGE + a manual sliding time window is a reasonable fallback.

**Limitations:** GNNs need more data and tuning than tabular baselines; risk of overfitting given the label is already weak/noisy — worth reporting whether GNN improvement over Random Forest is actually statistically meaningful, not just numerically higher.

---

## 10. Explainability

- **SHAP** — attributes each baseline model's prediction to individual features (e.g., "70% of this score came from heading variance"). Best for the tabular baselines.
- **GNNExplainer** — identifies which nodes/edges in the graph most influenced a given node's prediction (e.g., "this pedestrian's uncertainty score was driven by proximity to these 3 others").
- **Integrated Gradients** — attribution method for the MLP/sequence models, works well when inputs are continuous.
- **Attention visualization** — if using GAT/TGAT, the attention weights themselves can be visualized directly as an interpretability signal, no separate method needed.

These matter for this project specifically because the "not psychological confusion" framing is only credible if predictions can be traced back to concrete, physical kinematic causes.

---

## 11. Navigation Difficulty Heatmap

- **Spatial aggregation:** bin the scene into a grid; average (or take a robust statistic like median) of uncertainty scores of all trajectory points falling in each cell.
- **Grid generation:** cell size is a tradeoff — too fine and cells are sparse/noisy, too coarse and local hotspots get smoothed away. Start with a physical scale (~1–2 meters/cell) rather than an arbitrary pixel grid.
- **Color mapping:** sequential colormap (e.g., low = blue/green, high = red) to avoid implying a diverging "good/bad" binary.
- **Applications:** flagging signage placement issues, informing facility redesign, validating wayfinding interventions before/after comparison.
- **Potential improvements:** kernel density smoothing instead of hard binning; temporal heatmaps (does the confusing zone change by time of day/crowd density).

---

## 12. Research Gap

Trajectory prediction literature (Social-LSTM, Social-GAN, Trajectron++, Y-Net) focuses on *predicting where people go next*, assuming trajectories are goal-directed and largely rational. This body of work does not ask *how well a person is navigating* — it treats all movement as equally "confident."

Separately, human wayfinding/space-syntax literature (mostly from architecture/urban planning) studies confusing spatial layouts but typically relies on surveys, simulations, or manual observation — not large-scale trajectory data mining.

**This project fills the gap** by bridging these two literatures: applying trajectory-prediction-style feature engineering and modeling (including GNNs) to the wayfinding-difficulty question, using weak supervision to avoid the labeling bottleneck that has kept these fields separate.

**Novelty:** to your knowledge (and stated as an assumption to verify via literature review before submission), no existing work defines a continuous, weakly-supervised "navigation uncertainty index" from public pedestrian trajectory datasets and aggregates it into an actionable spatial heatmap using graph-based interaction modeling.

---

## 13. Novel Contributions

1. A formal, reproducible definition of trajectory-based Navigation Uncertainty that avoids unfounded psychological claims.
2. A weak-supervision labeling pipeline (entropy-weighted, optionally regression-refined) applicable to any public trajectory dataset without manual confusion annotation.
3. A comparative evaluation of tabular, sequence, and graph-based models for predicting this score.
4. A social-context-aware (GNN-based) uncertainty model that separates crowd-induced slowdown from wayfinding-induced slowdown.
5. An explainable, spatially-aggregated navigation difficulty heatmap usable as a design/planning artifact.

---

## 14. Potential Limitations (Be Honest)

- **Dataset limitations:** SDD scenes are campus environments — findings may not generalize to airports, hospitals, or transit stations without new data.
- **Weak supervision limitations:** the label is a proxy; a high score guarantees kinematic irregularity, not confusion. This must never be overstated in the writeup.
- **Behavior ambiguity:** stopping/reversing can stem from social interaction, phone use, photography, fatigue — the model cannot distinguish cause, only correlate pattern.
- **Generalization:** models trained on one scene/dataset may not transfer to structurally different spaces (indoor vs. outdoor, different crowd densities).
- **Label quality:** since there's no independent ground truth, quantitative validation is inherently limited to proxy checks (small manual review subsets, known-confusing-intersection sanity checks).
- **Ethics:** even with careful framing, any system inferring "confusion-like" states from surveillance-style trajectory data raises privacy/surveillance concerns that should be explicitly addressed in an ethics section — recommend anonymized, aggregate-only outputs (heatmaps), never per-individual "confusion scores" in any real deployment.
- **False positives:** crowded but efficient spaces may trigger irregularity features (stopping due to crowding, not confusion) — density-normalization is a needed mitigation, not a solved problem.

---

## 15. Future Work

- Transformer-based trajectory models (e.g., trajectory transformers) as an alternative to LSTM/GNN sequence modeling.
- Vision-Language Models to incorporate scene semantics (signage visibility, layout complexity) alongside pure kinematics.
- Foundation models / pretrained trajectory embeddings for cross-dataset transfer.
- Extending to indoor navigation datasets (e.g., mall/airport-specific data, JRDB).
- Real-time deployment feasibility study (latency, streaming trajectory processing).
- Multi-camera / multi-view fusion for occlusion-robust trajectories.

---

## 16. Complete Pipeline Diagram

```
Raw Trajectory Dataset (SDD / ETH / UCY)
            │
            ▼
   [1] Preprocessing & Cleaning
            │
            ▼
   [2] Trajectory Segmentation (sliding windows)
            │
            ▼
   [3] Behavioral Feature Extraction
   (speed, accel, stops, heading var, turning,
    U-turns, path efficiency, oscillation, hesitation)
            │
            ▼
   [4] Navigation Uncertainty Index Generation
   (entropy weighting → optional regression refinement)
            │
            ├──────────────► [5] Baseline ML Models
            │                (RF / XGBoost / MLP / LinReg)
            │
            ├──────────────► [6] Future Uncertainty Prediction
            │                (sequence models on partial trajectories)
            │
            └──────────────► [7] Spatio-Temporal GNN
                             (TGN / GAT — social-context aware)
                                       │
                                       ▼
                             [8] Explainability
                             (SHAP / GNNExplainer / attention)
                                       │
                                       ▼
                         [9] Spatial Aggregation → Heatmap
                             (grid binning + smoothing)
                                       │
                                       ▼
                     Navigation Difficulty Heatmap (Final Output)
```

---

## 17. Folder Structure

```
navigation-uncertainty/
├── data/
│   ├── raw/                  # untouched dataset downloads
│   ├── processed/            # cleaned trajectories
│   └── segments/             # windowed segments + features
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── labeling/              # uncertainty index generation
│   ├── models/
│   │   ├── baselines/
│   │   ├── sequence/
│   │   └── gnn/
│   ├── explainability/
│   └── heatmap/
├── notebooks/                 # exploratory analysis
├── configs/                   # experiment configs (yaml)
├── results/
│   ├── metrics/
│   └── figures/
├── docs/                      # thesis chapters, notes
├── requirements.txt
└── README.md
```

---

## 18. Technology Stack

- **Language:** Python 3.10+
- **Core libraries:** numpy, pandas, scikit-learn
- **Baseline ML:** xgboost, scikit-learn (RandomForest, LinearRegression)
- **Deep learning:** PyTorch
- **GNN:** PyTorch Geometric (PyG) or DGL
- **Explainability:** shap, captum (Integrated Gradients), PyG's GNNExplainer
- **Visualization:** matplotlib, seaborn, plotly (for interactive heatmaps)
- **Dev environment:** Jupyter/VS Code, Git for version control
- **GPU requirements:** a single mid-range GPU (e.g., 8–12GB VRAM) is sufficient for GNN/sequence model training at this scale; baselines run fine on CPU

---

## 19. Project Timeline (Suggested, ~14 weeks)

| Week | Focus |
|---|---|
| 1–2 | Literature review, dataset download & exploration, finalize scope |
| 3 | Preprocessing pipeline + segmentation |
| 4 | Behavioral feature engineering |
| 5 | Uncertainty index design (entropy weighting) + validation subset review |
| 6 | Baseline ML models (RF, XGBoost, MLP, LinReg) + evaluation |
| 7 | Sequence model for future uncertainty prediction |
| 8–9 | Graph construction + GNN model implementation (TGN/GAT) |
| 10 | Model comparison, ablations, statistical validation |
| 11 | Explainability (SHAP, GNNExplainer, attention viz) |
| 12 | Heatmap generation + spatial validation |
| 13 | Write-up: methodology, results, limitations, ethics |
| 14 | Final polish, presentation, buffer for delays |

---

## 20. References (real, established works only)

**Trajectory prediction:**
- Alahi, A. et al. (2016). *Social LSTM: Human Trajectory Prediction in Crowded Spaces.* CVPR.
- Gupta, A. et al. (2018). *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks.* CVPR.
- Salzmann, T. et al. (2020). *Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data.* ECCV.
- Mangalam, K. et al. (2020). *It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction (Y-Net).* ECCV.

**Datasets:**
- Robicquet, A. et al. (2016). *Learning Social Etiquette: Human Trajectory Understanding In Crowded Scenes* (Stanford Drone Dataset). ECCV.
- Pellegrini, S. et al. (2009). *You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking* (ETH). ICCV.
- Lerner, A. et al. (2007). *Crowds by Example* (UCY). Eurographics.
- Bock, J. et al. (2020). *The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections.* IEEE IV.
- Martin-Martin, R. et al. (2021). *JRDB: A Dataset and Benchmark for Visual Perception for Navigation in Human Environments.* IEEE TPAMI.

**Weak supervision:**
- Mintz, M. et al. (2009). *Distant Supervision for Relation Extraction Without Labeled Data.* ACL-IJCNLP.
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB.

**Graph Neural Networks:**
- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Veličković, P. et al. (2018). *Graph Attention Networks.* ICLR.
- Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS.
- Xu, D. et al. (2020). *Inductive Representation Learning on Temporal Graphs (TGAT).* ICLR.
- Rossi, E. et al. (2020). *Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN).* ICML Workshop.
- Yan, S. et al. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (ST-GCN).* AAAI.

**Explainability:**
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.
- Ying, R. et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS.
- Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML.

---

### Quick note for your defense/viva
The single sentence to repeat if a panel member pushes on ethics: *"We measure kinematic irregularity in movement, not psychological state — the model has no access to, and makes no claims about, what a pedestrian was thinking."* Keep that line ready; it's the load-bearing statement of the entire project's scientific legitimacy.
