
# Generalized Statistical Thermodynamic Learning (GSTL)
## Core Framework Theory Manual, v2.66 Consolidated Edition

**Purpose.** This manual is a standalone, durable reference for the core GSTL framework. It is written to support three uses at once:

1. a college-level course or seminar,
2. a specification sheet for spec-driven AI development,
3. a formal symposium document describing the mathematical core and its theoretical implications.

It deliberately focuses on the **general GSTL theory**, not on any one implementation such as VAEs, transformers, diffusion models, stock models, Sudoku solvers, or repair agents. Specific model families are treated only as examples of the abstract framework.

---

## 0. Executive Orientation

GSTL stands for **Generalized Statistical Thermodynamic Learning**.

The central thesis is:

> **Central thesis.** GSTL studies learning systems whose prediction, uncertainty, geometry, action, repair, and data collection are governed by validity.

A standard learning system asks:

$$
\text{What is the most likely prediction?}
$$

GSTL asks the larger closed-loop question:

$$
\text{What prediction, action, repair, query, or fallback is valid under the system's contracts?}
$$

A GSTL learner is therefore not just a predictor. It is a validity-governed dynamical system:

$$
\text{data}
\rightarrow
\text{model}
\rightarrow
\text{candidate}
\rightarrow
\text{validity assessment}
\rightarrow
\text{action / repair / query / fallback}
\rightarrow
\text{updated model}.
$$

The mature protected core remains:

$$
G_{\mathrm{core}}=(L,M,\mathsf P,I).
$$

where:

- $L$ is the statistical learning object;
- $M$ is the metric, geometry, or learned relevance object;
- $\mathsf P$ is the posterior or uncertainty protocol;
- $I$ is the validity contract system.

The current v2.66 consolidated neural-operational extension is:

$$
G_{\mathrm{GSTL}}
=
(L,M,\mathsf P,I,\mathsf R,\mathsf K,\mathsf A,\mathcal C,\mathcal M,\mathcal D),
$$

where:

- $\mathsf R$ is the residual and learned validity system;
- $\mathsf K$ is calibration;
- $\mathsf A$ is action and policy governance;
- $\mathcal C$ is candidate generation;
- $\mathcal M$ is memory over failures, repairs, audits, and residual signatures;
- $\mathcal D$ is deployment, monitoring, and compute governance.

The scalar diagnostic heart of GSTL is the total residual:

$$
R_G(z)=\sum_{j=1}^J w_j r_j(z),
$$

and CoreValidity:

$$
\mathrm{CV}_G(z)=\exp(-R_G(z)).
$$

Higher CoreValidity means lower residual incompatibility with the declared contracts.

---

## 1. Notation and Symbol Table

This table fixes the notation used throughout the manual and is intended to make the document usable as a course text, symposium handout, and specification reference.

| Symbol | Meaning |
|---|---|
| $q_\theta(y\mid x)$ | Parametric predictive model or autoregressive generator |
| $E_{\mathrm{lik},\theta}(z)$ | Likelihood or fit energy |
| $R_G(z)$ | Aggregate GSTL validity residual |
| $\mathrm{CV}_G(z)$ | CoreValidity, typically $\exp(-R_G(z)/\tau_R)$ or $\exp(-R_G(z))$ when $\tau_R=1$ |
| $\tau_R$ | Residual temperature or scale parameter |
| $E_G(z)$ | Total GSTL energy |
| $\mathcal F_G(q)$ | GSTL free-energy functional |
| $\pi_G$ | GSTL policy, selector, or thermodynamic action rule |
| $\mathcal V$ | Validity presheaf or sheaf |
| $\rho_{U,V}$ | Restriction map from region $U$ to subregion $V\subseteq U$ |
| $R_{\mathrm{glue}}$ | Sheaf gluing residual |
| $\mathbf{Val}$ | Category or 2-category of validity interfaces |
| $F:I\to I'$ | Contract transformation or interface morphism |
| $\phi_t$ | Learning state, including parameters, memory, and diagnostics |
| $J_G(x,y)$ | GSTL sequence/action score |
| $J_{\mathrm{repair}}(z,a)$ | Counterfactual repair score |
| $\widehat p_{\mathrm{valid}}(z)$ | Calibrated validity probability |
| $\widehat{\mathrm{CV}}_G(z)$ | Learned or calibrated CoreValidity estimate |
| $\mathcal C_\theta(x)$ | Candidate set generated for input $x$ |
| $\mathcal M$ | Memory over failures, residual signatures, repairs, and audit traces |
| $\sigma_E(z)$ | Multi-critic ensemble disagreement |
| $U_{\mathrm{pred}},U_{\mathrm{geom}},U_{\mathrm{contract}}$ | Predictive, geometric, and contract uncertainty channels |
| $\Psi$ | Aggregator for multi-channel uncertainty |

---

## 2. Claim-Status Discipline

GSTL is useful only if it clearly separates definitions, conditional theorems, diagnostics, approximations, and hypotheses.

### 2.1 Definitions

The following are definitions inside the GSTL schema:

- the protected core $G_{\mathrm{core}}=(L,M,\mathsf P,I)$;
- residual contracts $r_j$;
- total residual $R_G$;
- CoreValidity $\mathrm{CV}_G=\exp(-R_G)$;
- candidate sets $\mathcal C_\theta(x)$;
- sequence score $J_G$;
- validity calibration functions;
- deployment certificates;
- cascade and ensemble certificates.

### 2.2 Conditional theorems

Theorems hold under explicit assumptions. Examples:

- CoreValidity monotonicity under contract-nonexpansive morphisms;
- candidate coverage bounds selection success;
- ensemble veto blocks candidates under disagreement assumptions;
- cascade compute is lower than full compute when early stopping occurs;
- selected-action calibration differs from candidate calibration under selection shift;
- conservative policy improvement holds under support and uncertainty assumptions.

### 2.3 Diagnostics

Diagnostics are measurable quantities, not universal truths:

- Expected Calibration Error (ECE), Brier score, NLL;
- Goodhart gap;
- residual energy;
- support width;
- uncertainty traces;
- non-collapse certificate;
- metric witness error;
- glue residual;
- audit discovery rate;
- fallback utility.

### 2.4 Implementation approximations

Many practical objects approximate ideal theory:

- learned metrics approximate curvature;
- residual networks approximate residual contracts;
- ensembles approximate epistemic uncertainty;
- calibration maps approximate conditional validity probabilities;
- cascades approximate compute-optimal validation.

### 2.5 Non-claims

GSTL does **not** claim:

- universal convergence;
- no overfitting for arbitrary learners;
- guaranteed superiority over maximum likelihood on every benchmark;
- non-vacuous PAC-Bayes bounds without assumptions;
- physical thermodynamic laws unless a physical system is explicitly modeled;
- quantum advantage unless a quantum system is explicitly modeled;
- automatic generalization without some source of validity signal.

---

## 3. Mathematical Preliminaries

GSTL uses concepts from statistical learning, information geometry, Bayesian/posterior inference, calibration, stochastic control, thermodynamics, category theory, causal inference, and sequential decision theory.

### 3.1 Statistical learning object

Let:

$$
\mathcal X
$$

be an input space, and:

$$
\mathcal Y
$$

be an output or action space. A sample is:

$$
S=\{(x_i,y_i)\}_{i=1}^n\sim D^n.
$$

A model class is:

$$
\mathcal H=\{h_\theta:\mathcal X\to\mathcal Y\mid \theta\in\Theta\}.
$$

A loss is:

$$
\ell:\mathcal H\times \mathcal X\times \mathcal Y\to \mathbb R_{\ge0}.
$$

Empirical risk:

$$
\widehat R_S(\theta)=\frac1n\sum_{i=1}^n \ell(h_\theta,x_i,y_i).
$$

Population risk:

$$
R_D(\theta)=\mathbb E_{(x,y)\sim D}[\ell(h_\theta,x,y)].
$$

### 3.2 Geometry

A relevance geometry is a positive semidefinite or positive definite field:

$$
G_\phi(\theta)\succeq 0.
$$

It defines local norms:

$$
\|v\|_{G_\phi}^2=v^\top G_\phi v.
$$

Natural-gradient-like movement is:

$$
\dot\theta=-G_\phi(\theta)^{-1}\nabla_\theta \widehat R_S(\theta).
$$

GSTL does not require $G_\phi$ to be exactly Fisher information. It may be Fisher, empirical Fisher, GGN, Hessian approximation, learned metric, graph/sheaf Laplacian, or a task-specific preconditioner.

### 3.3 Posterior and uncertainty

A posterior or randomized learner is:

$$
Q\in\mathcal P(\Theta).
$$

A prior/reference measure is:

$$
P_0\in\mathcal P(\Theta).
$$

Predictive distribution:

$$
p_Q(y\mid x)=\mathbb E_{\theta\sim Q}[p_\theta(y\mid x)].
$$

GSTL decomposes uncertainty into at least three channels:

$$
U_G(z)=\big(U_{\mathrm{pred}}(z),U_{\mathrm{geom}}(z),U_{\mathrm{contract}}(z)\big).
$$

Predictive uncertainty concerns output ambiguity. Geometric uncertainty concerns unstable or ill-conditioned relevance geometry. Contract uncertainty concerns residual disagreement or invalidity.

### 3.4 Validity contracts

A contract is a typed requirement:

$$
C_j: \mathcal Z_j\to \{\mathrm{valid},\mathrm{invalid}\}
$$

or a soft residual:

$$
r_j:\mathcal Z_j\to\mathbb R_{\ge0}.
$$

A state $z$ may include predictions, actions, trajectories, repairs, memories, candidate sets, metrics, posterior samples, or deployment contexts.

A residual equals zero when the contract is satisfied:

$$
r_j(z)=0 \iff C_j(z)=\mathrm{valid}.
$$

---

## 4. The Protected Core

### 4.1 Definition: GSTL core

A GSTL core is:

$$
G_{\mathrm{core}}=(L,M,\mathsf P,I).
$$

where:

1. $L$ is a statistical learning object;
2. $M$ is a metric/relevance object;
3. $\mathsf P$ is a posterior/uncertainty protocol;
4. $I$ is a validity interface and contract system.

This definition emphasizes modularity across architectures: a GSTL core may be instantiated by a classical statistical model, a neural sequence model, an agent, a probabilistic program, a causal policy, or any system for which learning, relevance, uncertainty, and validity interfaces can be declared.

### 4.2 Definition: residual system

A residual system is a finite or countable family:

$$
\mathsf R=\{(r_j,w_j,\mathcal Z_j)\}_{j\in J}
$$

where:

$$
r_j:\mathcal Z_j\to\mathbb R_{\ge0},\qquad w_j\ge0.
$$

The total residual is:

$$
R_G(z)=\sum_{j\in J}w_j r_j(z).
$$

When the sum is infinite, assume convergence or define $R_G$ as a monotone limit of partial sums. More generally, GSTL may use a learned aggregator

$$
R_G(z)=A_\psi(r_1(z),\ldots,r_J(z)),
$$

where $A_\psi$ is a learned residual aggregator constrained to be monotone in safety-critical residual channels whenever the relevant contracts are non-negotiable.

### 4.3 Definition: CoreValidity

CoreValidity is:

$$
\mathrm{CV}_G(z)=\exp(-R_G(z)).
$$

Properties:

1. $0<\mathrm{CV}_G(z)\le1$.
2. $\mathrm{CV}_G(z)=1$ iff $R_G(z)=0$.
3. $R_G(z_1)\le R_G(z_2)$ iff $\mathrm{CV}_G(z_1)\ge \mathrm{CV}_G(z_2)$.

### 4.4 Proof of monotonicity

Suppose:

$$
R_G(z_1)\le R_G(z_2).
$$

Because $x\mapsto \exp(-x)$ is strictly decreasing on $\mathbb R$, we have:

$$
\exp(-R_G(z_1))\ge \exp(-R_G(z_2)).
$$

Thus:

$$
\mathrm{CV}_G(z_1)\ge \mathrm{CV}_G(z_2).
$$

This proves monotonicity of CoreValidity with respect to residual reduction.

---

## 5. Contract-Nonexpansive Transformations

### 5.1 Definition

A morphism between GSTL states or cores is **contract-nonexpansive** if:

$$
R_{G'}(Fz)\le R_G(z)
$$

for all admissible states $z$.

If instead:

$$
R_{G'}(Fz)\le R_G(z)+\epsilon,
$$

then $F$ is $\epsilon$-contract-nonexpansive.

### 5.2 Theorem: CoreValidity preservation

If $F$ is contract-nonexpansive, then:

$$
\mathrm{CV}_{G'}(Fz)\ge \mathrm{CV}_G(z).
$$

### Proof

By definition:

$$
R_{G'}(Fz)\le R_G(z).
$$

Apply the monotonicity result from Section 3.4:

$$
\exp(-R_{G'}(Fz))\ge\exp(-R_G(z)).
$$

Hence:

$$
\mathrm{CV}_{G'}(Fz)\ge \mathrm{CV}_G(z).
$$

### 5.3 Interpretation

This theorem is the formal backbone of GSTL: valid transformations are those that do not increase declared contract residuals. Training, repair, calibration, distillation, and deployment changes can all be judged by residual nonexpansion.

---

## 6. Categorical Organization

GSTL admits a categorical view.

### 6.1 Category of GSTL cores

Define a category:

$$
\mathbf{GSTL}_{\mathrm{core}}
$$

where:

- objects are GSTL cores;
- morphisms are contract-nonexpansive transformations.

Identity maps are contract-nonexpansive because:

$$
R_G(\mathrm{id}(z))=R_G(z).
$$

Composition is contract-nonexpansive because if:

$$
R_{G'}(Fz)\le R_G(z)
$$

and:

$$
R_{G''}(H(Fz))\le R_{G'}(Fz),
$$

then:

$$
R_{G''}(H\circ F(z))\le R_G(z).
$$

Thus contract-nonexpansive GSTL cores form a category.

### 6.2 Products and pullbacks

Independent composition can be modeled by products:

$$
G=G_1\times G_2.
$$

Residuals may combine as:

$$
R_G(z_1,z_2)=R_{G_1}(z_1)+R_{G_2}(z_2)
$$

or as a bottleneck:

$$
R_G(z_1,z_2)=\max(R_{G_1}(z_1),R_{G_2}(z_2)).
$$

Interface-constrained composition is modeled by pullbacks. If two modules share an interface $H$, then:

$$
G_1\times_H G_2
$$

contains pairs compatible over $H$.

### 6.3 Bottleneck theorem

If product validity is bottlenecked:

$$
\mathrm{CV}_{G_1\times G_2}(z_1,z_2)=\min(\mathrm{CV}_{G_1}(z_1),\mathrm{CV}_{G_2}(z_2)),
$$

then the composite validity cannot exceed the weakest component.

### Proof

By definition:

$$
\mathrm{CV}_{G_1\times G_2}=\min(\mathrm{CV}_{G_1},\mathrm{CV}_{G_2}).
$$

Therefore:

$$
\mathrm{CV}_{G_1\times G_2}\le \mathrm{CV}_{G_1},
\qquad
\mathrm{CV}_{G_1\times G_2}\le \mathrm{CV}_{G_2}.
$$

Thus the weakest component is the bottleneck.

### 6.4 Functors, adjunctions, and monads

A GSTL functor maps cores to enriched cores, for example:

$$
F:G\mapsto G^+
$$

where $G^+$ includes a new metric, residual, memory, or calibration module.

An adjunction:

$$
F\dashv U
$$

models enrichment and forgetting. The unit and counit residuals quantify the cost of adding or forgetting structure.

A monad:

$$
T=UF
$$

models repeated free generation, repair, normalization, or memory-augmented closure.

The monadic residual law is typically of the form:

$$
R_G(\mu_z(TTz))\le R_G(Tz)+\epsilon_\mu,
$$

where $\mu$ is multiplication/normalization.

---

## 7. Sheaf-Theoretic Formalization of Validity

Many validity failures are local before they are global. A proof can fail at one inference step, a generated program can fail at a function boundary, a legal document can fail at one clause, and a sequence can become inconsistent within a short span. GSTL formalizes this locality using presheaves and sheaves.

Let $(X,\mathcal T)$ be a topological space or combinatorial site whose open sets represent outputs, traces, patches, sequence spans, graph neighborhoods, proof lines, or agent states.

### 7.1 Definition: validity presheaf

A validity presheaf $\mathcal V$ assigns to each open set $U\subseteq X$ a set $\mathcal V(U)$ of locally valid assignments on $U$, together with restriction maps

$$
\rho_{U,V}:\mathcal V(U)\to\mathcal V(V)
$$

for every inclusion $V\subseteq U$, satisfying the usual presheaf axioms:

$$
\rho_{U,U}=\mathrm{id},
\qquad
\rho_{V,W}\circ \rho_{U,V}=\rho_{U,W}.
$$

### 7.2 Definition: validity sheaf

A presheaf $\mathcal V$ is a sheaf if, for every open cover $\{U_i\}_{i\in A}$ of an open set $U$ and every family of sections $s_i\in\mathcal V(U_i)$ that agree on overlaps,

$$
\rho_{U_i,\,U_i\cap U_j}(s_i)
=
\rho_{U_j,\,U_i\cap U_j}(s_j),
$$

there exists a unique global section $s\in\mathcal V(U)$ such that

$$
\rho_{U,U_i}(s)=s_i
$$

for all $i$.

In practical machine learning settings, requiring exact satisfaction of the sheaf gluing axiom is typically too rigid. GSTL therefore works primarily with soft sheaves, where compatibility is measured quantitatively through residuals rather than enforced exactly.

### 7.3 Overlap and gluing residuals

For local sections $s_i\in\mathcal V(U_i)$, define the overlap residual

$$
R_{ij}(s_i,s_j)
:=
d_{ij}\Big(
\rho_{U_i,\,U_i\cap U_j}(s_i),
\rho_{U_j,\,U_i\cap U_j}(s_j)
\Big)^2,
$$

where $d_{ij}$ is a metric on assignments over the overlap $U_i\cap U_j$. The sheaf gluing residual is then

$$
R_{\mathrm{glue}}(\{s_i\})
:=
\sum_{i<j:\,U_i\cap U_j\neq\emptyset}
a_{ij}R_{ij}(s_i,s_j),
\qquad a_{ij}\ge0.
$$

### 7.4 Definition: GSTL sheaf residual

A sheaf-aware GSTL residual takes the form

$$
R_G(z)
=
R_{\mathrm{local}}(z)
+
\lambda_{\mathrm{glue}}R_{\mathrm{glue}}(z)
+
\lambda_{\mathrm{global}}R_{\mathrm{global}}(z),
$$

where the local term scores patch validity, the gluing term scores compatibility across overlaps, and the global term scores end-to-end task validity.

### 7.5 Localized repair principle

The sheaf perspective is powerful because it yields a principled theory of localized repair: instead of regenerating an entire output, the system can identify the smallest inconsistent patches or overlaps and repair only those regions.

A localized repair operator may be written:

$$
\mathrm{Repair}_{\mathrm{loc}}(z)
=
\arg\min_{z'}
\left[
R_{\mathrm{local}}(z')
+
\lambda_{\mathrm{glue}}R_{\mathrm{glue}}(z')
+
C(z,z')
\right],
$$

where $C(z,z')$ penalizes unnecessary deviation from the original candidate.

---

## 8. Geometry and Relevance

### 8.1 Metric object

The GSTL metric object is:

$$
M=(G_\phi,T_\star,\mathcal A_M,R_M).
$$

where:

- $G_\phi$ is the learned metric/preconditioner;
- $T_\star$ is a target curvature or witness;
- $\mathcal A_M$ is an admissibility class;
- $R_M$ is a metric residual.

Metric residual example:

$$
R_M(\phi)=\|\log G_\phi-\log T_\star\|_F^2.
$$

or diagonal form:

$$
R_M(\phi)=\|\log \mathrm{diag}(G_\phi)-\log \mathrm{diag}(T_\star)\|_2^2.
$$

### 8.2 Natural-gradient interpretation

If $G_\phi$ approximates Fisher information, GSTL descent resembles natural gradient:

$$
\theta_{t+1}=\theta_t-\eta G_\phi(\theta_t)^{-1}\nabla \widehat R_S(\theta_t).
$$

GSTL does not require exact information geometry. It requires a declared relevance geometry and a measurable residual that tracks whether the geometry is useful or stable.

### 8.3 Geometric uncertainty

Metric uncertainty can be diagnosed by:

$$
U_{\mathrm{geom}}(z)=\log\left(1+\mathrm{tr}\big((G_\phi(z)+\epsilon I)^{-1}\big)\right).
$$

This is a sensitivity proxy. Large inverse trace suggests directions of low curvature or high sensitivity.

---

## 9. Posterior and Uncertainty Protocol

### 9.1 Posterior object

The posterior protocol is:

$$
\mathsf P=(P_0,Q,\Gamma,B).
$$

where:

- $P_0$ is a prior/reference;
- $Q$ is a posterior or randomized learner;
- $\Gamma$ is a posterior realization map;
- $B$ is a bound or diagnostic functor.

### 9.2 PAC-Bayes-style diagnostic

A common diagnostic is:

$$
\mathrm{PB}_{\mathrm{proxy}}
=
\widehat R_S(Q)+\lambda\mathrm{KL}(Q\|P_0).
$$

GSTL treats this as a proxy unless all assumptions for a PAC-Bayes theorem are explicitly declared.

### 9.3 Tri-channel uncertainty

GSTL uses:

$$
U_G(z)=\Psi\big(U_{\mathrm{pred}},U_{\mathrm{geom}},U_{\mathrm{contract}}\big).
$$

Predictive uncertainty:

$$
U_{\mathrm{pred}}(x)=H[p_Q(y\mid x)].
$$

Geometric uncertainty:

$$
U_{\mathrm{geom}}(z)=\log(1+\mathrm{tr}(G^{-1})).
$$

Contract uncertainty:

$$
U_{\mathrm{contract}}(z)=R_G(z)
$$

or a calibrated transformation of residual disagreement.

---

## 10. Validity, Calibration, and Reliability

### 10.1 Learned validity

A learned validity critic is:

$$
s_\psi(z)\in\mathbb R.
$$

A calibrated validity estimate is:

$$
\widehat{\mathrm{CV}}_G(z)=\kappa(s_\psi(z)).
$$

The target is:

$$
\widehat{\mathrm{CV}}_G(z)\approx \Pr[v(z)=1\mid z].
$$

### 10.2 Candidate-level calibration

Candidate calibration asks:

$$
\widehat{\mathrm{CV}}(z)\approx \Pr[v(z)=1\mid z].
$$

Expected Calibration Error:

$$
\mathrm{ECE}=
\sum_b\frac{|B_b|}{n}
\left|
\mathrm{acc}(B_b)-\mathrm{conf}(B_b)
\right|.
$$

### 10.3 Selected-action calibration

Let:

$$
z^*=\pi_G(x).
$$

Selected-action calibration asks:

$$
\widehat{\mathrm{CV}}(z^*)\approx \Pr[v(z^*)=1\mid z^*=\pi_G(x)].
$$

### 10.4 Theorem: selection can break calibration

Even if $\widehat{\mathrm{CV}}(z)$ is calibrated on the candidate distribution, it need not be calibrated on the selected-action distribution.

### Proof

Candidate calibration is calibration under distribution $P(z)$.

Selection induces:

$$
P_{\pi}(z)=P(z\mid z=\pi_G(x)).
$$

In general:

$$
P_{\pi}(z)\ne P(z).
$$

Calibration under $P$ does not imply calibration under an arbitrary conditional distribution $P_\pi$ unless additional invariance assumptions hold. Therefore selected-action calibration must be measured separately.

---

## 11. Candidate Generation and Coverage

### 11.1 Candidate set

A generator produces candidates:

$$
\mathcal C_\theta(x)=\{z^{(1)},\ldots,z^{(K)}\}.
$$

For sequences:

$$
z=(x,y),
\qquad
q_\theta(y\mid x)=\prod_t q_\theta(y_t\mid x,y_{<t}).
$$

### 11.2 Candidate coverage

Coverage is:

$$
\mathrm{Cov}_{G,K}(x)=\mathbf 1[\exists y\in \mathcal C_{\theta,K}(x):v(x,y)=1].
$$

### 11.3 Coverage theorem

For any selector:

$$
\pi(x)\in \mathcal C_{\theta,K}(x),
$$

we have:

$$
\mathrm{Success}_{\pi,K}(x)\le \mathrm{Cov}_{G,K}(x).
$$

### Proof

If $\mathrm{Cov}_{G,K}(x)=0$, then no candidate in $\mathcal C_{\theta,K}(x)$ is valid. Since $\pi(x)$ must choose from that set, $\pi(x)$ is invalid. Thus success is zero. If coverage is one, success is at most one. Hence success is bounded by coverage.

### 11.4 Implication

GSTL has two jobs:

1. improve candidate coverage;
2. rank covered candidates by validity.

No reranker can select a valid candidate that was never generated.

---

## 12. Full Sequence-Space Ranking

### 12.1 Autoregressive sequence likelihood

For output sequence:

$$
y=(y_1,\ldots,y_T),
$$

likelihood is:

$$
\log q_\theta(y\mid x)=\sum_{t=1}^T\log q_\theta(y_t\mid x,y_{<t}).
$$

### 12.2 GSTL sequence score

GSTL ranks full sequences by:

$$
J_G(x,y)=\log q_\theta(y\mid x)-\beta R_G(x,y).
$$

or:

$$
J_G(x,y)=\log q_\theta(y\mid x)+\lambda_{\mathrm{CV}}\log \widehat{\mathrm{CV}}_G(x,y).
$$

### 12.3 Soft ranking condition

GSTL prefers $y_a$ to $y_b$ iff:

$$
\log\frac{q_\theta(y_a\mid x)}{q_\theta(y_b\mid x)}
>
\beta\left(R_G(x,y_a)-R_G(x,y_b)\right).
$$

Equivalently, validity can overcome likelihood disadvantage when residual improvement is large enough.

### 12.4 Lexicographic ranking

For hard validity constraints:

$$
y^*_{\mathrm{lex}}
=
\mathrm{lexmax}_{y\in\mathcal C_\theta(x)}
\left(
\mathbf 1[R_G(x,y)\le \tau_R],
-R_G(x,y),
\log q_\theta(y\mid x)
\right).
$$

This is appropriate when contracts are non-negotiable.

---

## 13. Thermodynamic Core

### 13.1 Energy

GSTL energy can be written:

$$
E_G(z)=E_{\mathrm{fit}}(z)+\beta R_G(z)+\gamma U_G(z)+\eta C(z).
$$

where:

- $E_{\mathrm{fit}}$ is loss or negative likelihood;
- $R_G$ is validity residual energy;
- $U_G$ is uncertainty energy;
- $C$ is action, repair, query, or compute cost.

### 13.2 Free energy

For distribution $q$ over states:

$$
\mathcal F_G(q)=\mathbb E_{z\sim q}[E_G(z)]-T H(q).
$$

### 13.3 Derivation of Gibbs form

Minimize:

$$
\mathcal F(q)=\sum_z q(z)E(z)+T\sum_z q(z)\log q(z)
$$

subject to:

$$
\sum_z q(z)=1.
$$

Lagrangian:

$$
\mathcal L(q,\lambda)=\sum_z q(z)E(z)+T\sum_z q(z)\log q(z)+\lambda\left(\sum_z q(z)-1\right).
$$

Set derivative to zero:

$$
E(z)+T(1+\log q(z))+\lambda=0.
$$

Thus:

$$
q(z)\propto \exp(-E(z)/T).
$$

For GSTL:

$$
q_G(z)\propto \exp(-E_G(z)/T).
$$


### 13.4 Relative free-energy dissipation rate

When GSTL is used to analyze a learning trajectory $\phi_t$, define the relative free-energy dissipation rate as

$$
\mathcal D_G(\phi_t)
:=
-\frac{d}{dt}\mathcal F_G(\phi_t)
$$

whenever the derivative exists, or by the discrete proxy

$$
\widehat{\mathcal D}_G(t)
:=
\mathcal F_G(\phi_t)-\mathcal F_G(\phi_{t+1}).
$$

This is a diagnostic unless the stochastic process, reference measure, and thermodynamic assumptions are explicitly specified. The term is useful because it names the quantity that tracks whether learning dissipates validity-residual energy over time.


### 13.5 Interpretation

GSTL generalizes energy-based modeling by adding declared validity residuals, uncertainty, and action cost. It is not merely an energy model; it is a validity-governed thermodynamic control framework.

---

## 14. Action, Repair, Query, and Fallback

### 14.1 Action policy

A GSTL policy is:

$$
\pi_G:\mathcal Z\to\mathcal A.
$$

It may choose:

- accept;
- reject;
- repair;
- query;
- fallback;
- abstain;
- collect data;
- intervene.

### 14.2 Validity-governed action

A soft action objective is:

$$
a^*=\arg\max_a
\left[
U(a,z)+\lambda_{\mathrm{CV}}\log \widehat{\mathrm{CV}}_G(a,z)-C(a,z)
\right].
$$

A hard constrained action is:

$$
a^*=\arg\max_a U(a,z)
\quad\text{subject to}\quad
R_G(a,z)\le\tau_R.
$$

### 14.3 Repair operators

A repair operator is:

$$
\rho:\mathcal Z\to\mathcal Z.
$$

It is successful if:

$$
R_G(\rho(z))<R_G(z).
$$

A learned repair generator produces:

$$
\mathcal C_{\mathrm{repair}}(z)=\{z_1^+,\ldots,z_K^+\}.
$$

and selects:

$$
z^*=\arg\max_{z^+}
\left[
\log p(z^+\mid z)+\lambda\log \widehat{\mathrm{CV}}_G(z^+)-C(z,z^+)
\right].
$$

### 14.4 Jump inference

A prefix state is:

$$
s_t=(x,y_{<t}).
$$

Likelihood token:

$$
y_t^{\mathrm{MLE}}=\arg\max_a \log q_\theta(a\mid s_t).
$$

GSTL jump token:

$$
y_t^{\mathrm{GSTL}}=
\arg\max_a
\left[
\log q_\theta(a\mid s_t)-\beta R_G(s_t,a)
\right].
$$

Jump inference is triggered when:

$$
R_G(s_t,y_t^{\mathrm{MLE}})>\tau_R
$$

or when validity gain exceeds likelihood loss.

---

## 15. Memory and Retrieval

### 15.1 Residual signatures

A residual signature is:

$$
\mathcal V_G(z)=(r_1(z),\ldots,r_J(z)).
$$

### 15.2 Memory

GSTL memory stores:

$$
\mathcal M=\{(z_i,\mathcal V_G(z_i),a_i,\rho_i,\Delta R_i)\}_{i=1}^N.
$$

### 15.3 Retrieval

Retrieve similar failures by:

$$
\mathrm{Retrieve}(z)=\arg\min_i d(\mathcal V_G(z),\mathcal V_G(z_i)).
$$

### 15.4 Memory robustness

If retrieval is noisy, GSTL must estimate:

$$
\Pr[\rho_i\text{ succeeds}\mid \mathcal V_G(z),\mathcal V_G(z_i)].
$$

and fallback when repair confidence is low.

---

## 16. Goodhart, Adversarial Validity, and Red-Team Audit

### 16.1 Learned validity can be exploited

If a generator optimizes:

$$
q_\theta(z)\propto \exp(\ell_\theta(z)+\beta \widehat{\mathrm{CV}}_\psi(z)),
$$

then it may find high predicted-validity but invalid candidates.

### 16.2 Goodhart gap

Define:

$$
\Delta_G
=
\mathbb E_{z\sim q_\theta}[\widehat{\mathrm{CV}}_\psi(z)]
-
\mathbb E_{z\sim q_\theta}[v(z)].
$$

Large positive $\Delta_G$ indicates validity critic exploitation.

### 16.3 Red-team audit

Audit acquisition:

$$
\alpha_{\mathrm{audit}}(z)=
\widehat{\mathrm{CV}}_\psi(z)
\cdot
\mathrm{Risk}_{\mathrm{OOD}}(z)
\cdot
\mathrm{Exploitability}(z).
$$

This targets high-confidence invalid candidates, not merely uncertain examples.

### 16.4 Iterative audit loop

$$
\mathcal D_0
\rightarrow
\widehat{\mathrm{CV}}_\psi
\rightarrow
\alpha_{\mathrm{audit}}
\rightarrow
\mathcal A_K
\rightarrow
\mathcal D_1
\rightarrow
\widehat{\mathrm{CV}}_{\psi^+}.
$$

### 16.5 Firewall certificate

Adversarial risk:

$$
A_G(q_\theta,\psi)=\sup_{z:v(z)=0,q_\theta(z)>0}\widehat{\mathrm{CV}}_\psi(z).
$$

Firewall:

$$
\mathcal C_{\mathrm{fire}}=\mathbf 1[A_G(q_\theta,\psi)\le \epsilon_A].
$$

Deployment is allowed only if:

$$
\mathcal C_{\mathrm{fire}}=1.
$$

---

## 17. Multi-Critic Validity Ensembles

### 17.1 Ensemble

A critic ensemble is:

$$
\mathcal E_G(z)=\{c_1(z),\ldots,c_M(z)\}.
$$

Mean:

$$
\mu_E(z)=\frac1M\sum_{m=1}^M c_m(z).
$$

Disagreement:

$$
\sigma_E(z)=\left(\frac1M\sum_{m=1}^M(c_m(z)-\mu_E(z))^2\right)^{1/2}.
$$

Minimum critic:

$$
c_{\min}(z)=\min_m c_m(z).
$$

### 17.2 Certificate

Certified validity requires:

$$
\mu_E(z)\ge\tau_\mu,
$$

$$
\sigma_E(z)\le\tau_\sigma,
$$

$$
c_{\min}(z)\ge\tau_{\min}.
$$

### 17.3 Ensemble blocking theorem

If a spoof fools only a strict subset of critics, then disagreement is nonzero and the minimum critic is low. Sufficiently strict thresholds block the spoof.

### Proof

Suppose $c_i(z)$ is high for some critics and low for at least one critic. Then:

$$
c_{\min}(z)<\tau_{\min}
$$

for appropriate $\tau_{\min}$, or:

$$
\sigma_E(z)>\tau_\sigma
$$

for appropriate $\tau_\sigma$. Thus the certificate fails.

---

## 18. Distillation, Cascades, and Batched Certification

### 18.1 Distilled validity

A student validity model is:

$$
d_\eta(z)\approx \mu_E(z).
$$

Conservative distillation loss:

$$
\mathcal L_{\mathrm{distill}}
=
\mathbb E[(d_\eta(z)-\mu_E(z))^2]
+
\lambda_{\mathrm{veto}}\mathbb E[(d_\eta(z)-c_{\min}(z))_+^2].
$$

### 18.2 Selective fallback

$$
\pi(z)=
\begin{cases}
\pi_d(z),& d_\eta(z)\ge\tau_d,\ u_\eta(z)\le\tau_u,\\
\pi_E(z),&\text{otherwise.}
\end{cases}
$$

### 18.3 Cascaded certification

Stages:

$$
(c_1,c_2,\ldots,c_K)
$$

with increasing cost.

Candidate batch:

$$
B_1=B.
$$

Escalation:

$$
B_{k+1}=\{z\in B_k:\mathrm{Cert}_k(z)=0\}.
$$

Scalar compute:

$$
C(B)=\sum_{k=1}^K c_k|B_k|.
$$

Latency:

$$
T(B)=\sum_{k=1}^K(\alpha_k+\beta_k|B_k|^{\rho_k}).
$$

### 18.4 Cost dominance theorem

If at least one candidate certifies before the final stage, and all costs are positive, then cascade compute is lower than evaluating all stages on all candidates.

### Proof

Full compute:

$$
C_{\mathrm{full}}=|B|\sum_k c_k.
$$

Cascade compute:

$$
C_{\mathrm{cas}}=\sum_k c_k|B_k|.
$$

Since $B_k\subseteq B$ for all $k$, $|B_k|\le |B|$. If at least one early stop occurs, then for some later $k$, $|B_k|<|B|$. With positive costs:

$$
C_{\mathrm{cas}}<C_{\mathrm{full}}.
$$

---

## 19. Automatic Validity Learning

### 19.1 Concept

Automatic validity learning means residual structure is inferred from data, not only written by hand.

Pipeline:

$$
\text{data}\rightarrow \text{patterns}\rightarrow \text{implicit residuals}\rightarrow \text{validity credit}\rightarrow \text{better model}.
$$

### 19.2 Sources of implicit validity

- input-output consistency;
- correction traces;
- contrastive candidates;
- repair success/failure;
- repeated generic failure modes;
- causal intervention outcomes;
- critic disagreement;
- support gaps;
- temporal coherence;
- sheaf/gluing failures.

### 19.3 Learned residual model

$$
\widehat R_\psi(z)\approx R_G(z).
$$

A learned validity model:

$$
\widehat{\mathrm{CV}}_\psi(z)\approx \Pr[v(z)=1\mid z].
$$

### 19.4 Stronger prediction hypothesis

If validity reveals invariant structure hidden by biased likelihood, then:

$$
\mathrm{TruePrediction}_{\mathrm{GSTL}}
>
\mathrm{TruePrediction}_{\mathrm{MLE}}.
$$

This is an experimental hypothesis, not a universal theorem.

---

## 20. Off-Policy Evaluation, Support, and Safe Improvement

### 20.1 Logged data

Logged data:

$$
\mathcal D=\{(x_i,a_i,r_i,p_i)\}_{i=1}^n
$$

where $p_i$ is behavior propensity.

### 20.2 Importance weighting

For target policy $\pi$:

$$
\widehat V_{\mathrm{IPS}}(\pi)=\frac1n\sum_i \frac{\pi(a_i\mid x_i)}{p_i}r_i.
$$

### 20.3 Support residual

$$
R_{\mathrm{support}}(x,a)=\left[\tau_p-p_b(a\mid x)\right]_+.
$$

### 20.4 Conservative improvement

GSTL policy improvement is constrained by support:

$$
\pi_{\mathrm{new}}=\arg\max_\pi \widehat V_{\mathrm{DR}}(\pi)-\lambda R_{\mathrm{support}}(\pi).
$$

### 20.5 Hidden confounding and partial identification

When propensities or covariates are incomplete, GSTL reports intervals:

$$
V(\pi)\in [V^-(\Gamma),V^+(\Gamma)].
$$

A policy is safely better only if:

$$
V^-_{\mathrm{new}}(\Gamma)>V^+_{\mathrm{old}}(\Gamma).
$$

---

## 21. Causal Intervention and Remediation Graphs

### 21.1 Residual interaction graph

Residuals may interact:

$$
\mathcal G_R=(\{r_j\},E_R).
$$

An edge means changing one residual affects another.

### 21.2 Intervention effect

An intervention $a$ has effect:

$$
\Delta R_j(a)=R_j(z)-R_j(T_a z).
$$

### 21.3 Counterfactual remediation

GSTL chooses interventions by predicted residual improvement:

$$
a^*=\arg\max_a \mathbb E[\Delta R_G(a)]-C(a)-\lambda U(a).
$$

### 21.4 Causal caution

Correlation among residuals is not causation. GSTL must distinguish:

- observational residual association;
- intervention-tested residual effect;
- counterfactual repair prediction.

---

## 22. Broader Theoretical Context

### 22.1 Relation to statistical learning theory

GSTL extends empirical risk minimization with validity residuals, posterior uncertainty, and action constraints.

ERM asks:

$$
\min_\theta \widehat R_S(\theta).
$$

GSTL asks:

$$
\min_\theta \widehat R_S(\theta)+\lambda R_G(\theta)+\gamma U_G(\theta).
$$

### 22.2 Relation to PAC-Bayes

GSTL can include PAC-Bayes terms but does not claim them automatically. A valid PAC-Bayes theorem requires declared prior, posterior, loss boundedness, and probability statement.

### 22.3 Relation to information geometry

GSTL's metric object generalizes natural gradient and relevance geometry. The metric defines which distinctions matter for learning and validity.

### 22.4 Relation to thermodynamics

GSTL uses thermodynamic form as an organizing principle: energy, entropy, free energy, temperature, and irreversible repair. It is not automatically a physical thermodynamic law.

### 22.5 Relation to category theory

GSTL uses categories to organize compositional learning systems with contract-controlled transformations.

### 22.6 Relation to control theory

GSTL is a feedback control system:

$$
\text{state}\rightarrow \text{diagnostic}\rightarrow \text{action}\rightarrow \text{state update}.
$$

### 22.7 Relation to causal inference

GSTL incorporates causal intervention when repairs or policies change the world, not merely predictions.

### 22.8 Relation to transformers and diffusion

For transformers, GSTL governs sequence ranking, calibration, jump inference, and candidate coverage.

For diffusion models, GSTL will govern denoising trajectories:

$$
z_T\to z_{T-1}\to\cdots\to z_0
$$

with validity residuals at intermediate states.


### 22.9 Relation to quantum extensions

GSTL may be extended with quantum language only when there is a precise categorical, statistical, or physical justification. A true quantum GSTL model must specify density operators, measurements or POVMs, quantum Fisher or another valid quantum metric, and CPTP or Lindblad-style dynamics when dynamics are claimed. Without those declarations, quantum terminology remains metaphorical and should not be used as a theorem or performance claim.

---

## 23. Implementation Specification

A GSTL implementation should declare the following modules.

### 23.1 Core object

```text
GSTLCore:
  L: statistical learning object
  M: metric/relevance object
  P: posterior/uncertainty protocol
  I: validity contract system
```

### 23.2 Residual interface

```text
Residual:
  name: string
  domain: state type
  value(z): float >= 0
  weight: float >= 0
  status: hard | soft | diagnostic
```

### 23.3 Validity module

```text
ValidityModule:
  residual_vector(z) -> R^J
  total_residual(z) -> R_+
  core_validity(z) -> (0,1]
  calibrated_validity(z) -> [0,1]
```

### 23.4 Candidate module

```text
CandidateGenerator:
  generate(x, K) -> {z_1,...,z_K}
  coverage_estimate(x) -> [0,1]
```

### 23.5 Action module

```text
ActionPolicy:
  select(candidates, validity, utility, cost) -> action
  repair(z) -> z_plus
  query(z) -> query
  fallback(z) -> safe_action
```

### 23.6 Audit module

```text
Audit:
  find_high_confidence_invalids()
  estimate_goodhart_gap()
  update_validity_memory()
```

### 23.7 Cascade module

```text
Cascade:
  stage_1(z)
  stage_2(z)
  ...
  full_certificate(z)
  stop_or_escalate(z)
```

---

## 24. Diagnostics Catalogue

Every GSTL system should report a subset of:

### Adequacy

- validation loss;
- task accuracy;
- sequence validity;
- repair success;
- policy value.

### Validity

- residual energy;
- CoreValidity;
- valid rate;
- hallucination/spoof rate;
- selected-action validity.

### Calibration

- candidate ECE;
- selected-action ECE;
- Brier score;
- NLL;
- calibration by split.

### Geometry

- metric fit loss;
- condition number;
- inverse sensitivity trace;
- curvature approximation error.

### Uncertainty

- predictive entropy;
- epistemic variance;
- geometric uncertainty;
- contract uncertainty.

### Coverage

- candidate coverage;
- support width;
- OOD support risk.

### Goodhart and audit

- Goodhart gap;
- adversarial validity risk;
- audit discovery rate;
- red-team patch effect.

### Deployment

- fallback rate;
- compute cost;
- latency;
- throughput;
- abstention rate;
- unsafe deployment count.

---

## 25. Course Structure

A college course on GSTL can be taught in twelve modules.

1. Statistical learning and why likelihood is insufficient.
2. The GSTL protected core: $L,M,\mathsf P,I$.
3. Residuals, contracts, and CoreValidity.
4. Geometry, natural gradients, and relevance.
5. Posterior uncertainty and calibration.
6. Thermodynamic free energy and validity energy.
7. Category-theoretic composition of learning systems.
8. Candidate generation, coverage, and sequence-space ranking.
9. Repair, jump inference, memory, and fallback.
10. Goodhart, red-team audit, and ensemble validity.
11. Cascades, distillation, and scalable deployment.
12. Causal intervention, off-policy validity, and future extensions.

Suggested capstone: implement a small GSTL sequence learner and compare maximum likelihood against validity-governed sequence prediction under biased/noisy labels.

---

## 26. Symposium Abstract

Generalized Statistical Thermodynamic Learning is a framework for learning systems whose predictions, uncertainty, geometry, and actions are governed by explicit validity contracts. Its protected core is the quadruple $(L,M,\mathsf P,I)$: statistical learning object, relevance geometry, posterior uncertainty protocol, and validity interface system. GSTL defines residual contracts, aggregates them into total residual energy $R_G$, and induces CoreValidity $\mathrm{CV}_G=\exp(-R_G)$. The framework extends empirical risk minimization by treating validity, calibration, uncertainty, repair, querying, and deployment as first-class mathematical objects. It supports categorical composition, thermodynamic free-energy interpretation, candidate coverage bounds, calibrated sequence-space ranking, Goodhart-resistant validity critics, ensemble disagreement certificates, and cascaded compute-efficient validation. GSTL does not claim universal convergence or guaranteed superiority on raw likelihood. Its central operational hypothesis is that validity and uncertainty expose invariant structure that can improve true prediction, action, and repair under distribution shift.

---


## 27. Main Mathematical Contributions

The consolidated GSTL core contributes the following general mathematical objects:

1. **Validity-governed learning core.** A protected quadruple $G_{\mathrm{core}}=(L,M,\mathsf P,I)$ linking statistical learning, relevance geometry, posterior uncertainty, and validity contracts.
2. **Residual-to-validity bridge.** A total residual $R_G$ and CoreValidity $\mathrm{CV}_G$ that turn heterogeneous contract violations into a composable validity diagnostic.
3. **Contract-controlled morphisms.** A categorical language in which model transformations, compositions, hierarchies, adjunctions, and closures must preserve or explicitly bound residual growth.
4. **Local-to-global validity.** A sheaf-theoretic account of local compatibility, gluing residuals, and localized repair.
5. **Thermodynamic governance.** A free-energy perspective combining likelihood, validity residuals, uncertainty, entropy, and action or compute cost.
6. **Operational certification.** Calibration, ensemble disagreement, red-team audit, distillation, and cascaded validation mechanisms that convert validity from a diagnostic into a control principle.

---

## 28. Final Manifesto

A learning system should know not only what it predicts, but whether its prediction is valid, how confident its validity is, what residuals remain, whether it should repair, whether it should query, whether it should jump, whether it should fallback, and whether it should stop.

GSTL turns validity from a diagnostic into a control principle.

> **Final thesis.** GSTL is the theory of residual-governed learning systems whose prediction, geometry, uncertainty, memory, and action are organized by validity.
