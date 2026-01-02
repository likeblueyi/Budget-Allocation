# Budget Allocation and Its End-to-End Combinatorial Optimization Strategy
## Problem Introduction
**Problem Background:** Uncertainty-embedded budget allocation problems arise in contexts where nonprofit entities endeavor to propagate philanthropic information across a portfolio of websites, all while operating under the binding constraint of a fixed total budget.

**Prediction Phase:** Given the feature representation $\mathbf{x}^w$ pertaining to website $w$, the task consists of forecasting $\mathbf{y}^w$ namely, the likelihood that informational content hosted on website will successfully reach the intended customer base, as  below.
$  y^w=\mathcal{M}_\theta(\mathbf{x}^w). $

**Decision Phase:** The overarching objective is to maximize the expected count of users who are reached by at least one website within the set. Mathematically, this optimization challenge is formalized as below.
$$
\mathbf{v}^*(\mathbf{y}) = \arg\max_{\mathbf{v}} \frac{1}{N} \sum_{u=0}^{N} \left( 1 - \prod_{w=0}^{M} \left( 1 - \mathbf{v}^w \circ \mathbf{y}^{wu} \right) \right)
$$

$$
\text{subject to: }\sum_{w=0}^{M} \mathbf{v}^w\leq B 
$$

**Dataset and License:**
The data comes from Yahoo! Webscope Dataset (Yahoo. Yahoo! webscope dataset, 2007.) with labels $y_i$ for each user $i$. Access to this dataset is facilitated via Yahoo’s publicly accessible data repository, which is expressly designated for non-commercial utilization by academic researchers and scholars. The dataset adheres to Yahoo’s stringent data protection protocols, incorporating robust privacy safeguards. Any utilization of the data must strictly conform to the Data Sharing Agreement and the terms of use stipulated by Yahoo.

## Baseline Comparison Methods Introduction
**SPO**
In parallel, an alternative research strand has concentrated on adapting subgradient approximation methodologies, originally devised for continuous linear problems, to discrete-valued scenarios. Specifically, the SPO-relax method introduces a relaxation of the original discrete optimization problem and leverages the surrogate SPO+ loss function, first proposed in  Mandi et al. (2020). This loss formulation enables the utilization of subgradient-based updates within a backpropagation-compatible paradigm. Mathematically, the SPO-relax loss is defined as below.
$$
\mathcal{L}_{\text{spo}}(\mathbf{y},\hat{\mathbf{y}}) = -f\bigl(\mathbf{v}^*(2\hat{\mathbf{y}}-\mathbf{y}),2\hat{\mathbf{y}}-\mathbf{y}\bigr) + 2f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr) - f\bigl(\mathbf{v}^*(\mathbf{y}),\mathbf{y}\bigr)
$$

**NCE**
Mandi et al. (2022) t take \( \mathbb{S} \setminus \{\mathbf{v}^*(c)\} \) as negative examples and define a noise-contrastive estimation (NCE) loss, as below.
$$
\mathcal{L}_{\text{NCE}}(\hat{c}, c)=\frac{1}{|\mathbb{S}|}\sum_{\mathbf{v} \in \mathbb{S}}\left(f(\mathbf{v}^*(c), \hat{\mathbf{c}}) - f(\mathbf{v}, \hat{\mathbf{c}})\right)
$$
The novelty lies in the above formula being differentiable without solving the optimization problem. Moreover, if solutions in \( \mathbb{S}\) are optimal for arbitrary cost vectors, this approach is equivalent to training within a region of the convex hull of \( \mathbb{V} \).

**CpLayer**
Agrawal et al. (2019)  propose an approach to differentiate through disciplined convex programs (a subset of convex optimization problems used in domain-specific languages). Introducing disciplined parametrized programming (a subset of disciplined convex programming), they show every such program can be represented as composing an affine map from parameters to problem data, a solver, and an affine map from solver solution to original problem solution.

**Identity**
Sahoo et al. (2023)  propose a hyperparameter-free approach to embed discrete solvers as differentiable layers in deep learning. Prior methods (input perturbations, relaxation, etc.) have drawbacks like extra hyperparameters or compromised performance. Their work leverages the geometry of discrete solution spaces, treats solvers as negative identities in backpropagation, and uses generic regularization to avoid cost collapse. \textbf{I} is the identity matrix, and the gradient designed in their paper is shown as below.
$$
\frac{\partial \mathbf{v}}{\partial \mathbf{y}} = -\mathbf{I}
$$

**LODL and DFL**
Mandi et al. (2022) propose a novel approach that abandons surrogates entirely, instead learning loss functions tailored to task-specific information. Notably, theirs is the first method to fully replace the optimization component in decision-focused learning with an automatically learned loss. Key advantages include: (a) reliance only on a black-box oracle for solving the optimization problem, ensuring generalizability; (b) convexity by design, enabling straightforward optimization.

**Blackbox**
When confronted with the dilemma that the map from $\mathbb{C}\rightarrow\mathbb{V}$ is either non-differentiable or has vanishing gradients,  Poganˇci´c et al. (2019)  adopt a remarkably straightforward remedy: they approximate the gradient via linear interpolation. Their surrogate gradient construction is shown as below:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = \frac{1}{\lambda} \left[ \mathbf{v} \left( \hat{\mathbf{y}} + \lambda \frac{\partial L}{\partial \mathbf{v}} (\hat{\mathbf{v}}) \right) - \mathbf{v} (\hat{\mathbf{y}}) \right]
$$

**2-stage**
To ensure an equitable comparison, all end-to-end trainable models and the 2-stage baseline share an identical predictive backbone: a compact multi-layer perceptron (MLP).
Given an input feature vector $\mathbf{x}$, the predictor $\mathcal{M}$ is defined by the recursive relation:
$$
\mathbf{a}^{(1)} = \mathbf{x}
$$
$$
\mathbf{a}^{(i+1)} = \phi\!\bigl(\mathbf{W}^{(i)}\mathbf{a}^{(i)}+\mathbf{b}^{(i)}\bigr),  i=1,\dots,K-1
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{(K)}
$$
where $\mathbf{W}^{(i)}$ and $\mathbf{b}^{(i)}$ denote the weight matrix and bias vector of the $i$-th layer, respectively, and $\phi(\cdot)=\max(\cdot,0)$ is the ReLU activation.
Throughout the experiments we fix the depth at $K=3$ and the hidden dimension at $32$.

The 2-stage paradigm serves as the standard baseline whenever the coefficients of the downstream optimization task are uncertain and must be forecast.
A supervised predictor is trained on the pre-collected dataset $\mathcal{D}=\{(\mathbf{c}_i,\mathbf{y}_i)\}_{i=1}^{N}$ to minimize either the mean square error (MSE) loss as below.
$$
\mathcal{L}_{\text{MSE}}(\hat{\mathbf{y}},\mathbf{y}) = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{y}_i-\hat{\mathbf{y}}_i\|^2
$$
or the binary cross-entropy (BCE) loss as below:
$$
\mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}},\mathbf{y}) = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Bigr]
$$

At test time, the inferred coefficients $\hat{\mathbf{c}}=\mathcal{M}_\theta(\mathbf{x})$ are treated as deterministic inputs, after which an off-the-shelf solver is invoked to obtain the final decision.

Notably, the overall training objective in this 2-stage pipeline is entirely dictated by the \emph{prediction} loss (MSE or BCE); no task-specific decision loss is backpropagated.

## Experiments Results
Evaluation Metric: Regret. Regret is defined as follows. We hope that for a set of regret values (c, ĉ), the value is zero, or preferably, as small as possible. A smaller value indicates that the benefit of the post-prediction decision is closer to the benefit of the prior decision.
$regret (\mathbf{c},\hat{\mathbf{c}})=||f(\mathbf{z}^*(\mathbf{c}); \mathbf{c})-f(\mathbf{z}^*(\hat{\mathbf{c}}); \mathbf{c})||
$

#### Results on the test set:
 | Methods         | Relative Regret on Budget Allocation |
|-----------------|-------------------|
| 2-stage         | 20.332            |
| DFL             | 35.970            |
| Blackbox        | 26.905            |
| Identity        | 14.799            |
| CPLayer         | --                |
| SPO             | 5.559             |
| LODL            | 25.700            |
| NCE             | 9.979             |
| Org-LTR         | 5.742             |
| SAA-LTR (ours)  | $\textbf{\underline{4.259}}$ |

