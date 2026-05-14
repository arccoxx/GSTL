# GSTL — Generalized Statistical Thermodynamic Learning

> **Learning should not stop at prediction. A learning system should know when its inference is valid, when it is uncertain, when it should repair itself, when it should ask for more information, and when it should refuse to act.**

GSTL is a mathematical framework for **statistical learning of validity**.

Most machine learning asks:

> What is the most likely prediction?

GSTL asks a deeper question:

> Is this prediction valid enough to act on?

That shift changes the object of learning. The system is no longer just a predictor. It becomes a validity-governed closed loop:

```text
data → model → candidate prediction/action → validity check → repair/query/fallback/action → updated model
```

GSTL combines ideas from statistical learning, thermodynamics, information geometry, uncertainty, calibration, sheaf-style locality, action selection, repair, audit, and deployment governance into one core principle:

> **Validity is not an afterthought. Validity is a control signal.**

---

## 1. The Problem GSTL Tries to Solve

Modern models are powerful pattern learners. They can predict fluent text, classify images, forecast time series, generate plans, and propose actions.

But high probability is not the same as validity.

A model can produce something that is:

- statistically likely but semantically wrong,
- fluent but invalid,
- confident but unsupported,
- locally plausible but globally inconsistent,
- high scoring but unsafe to deploy,
- correct under training distribution but wrong under shift.

Maximum likelihood learning rewards what appears often. But what appears often is not always what is valid.

GSTL begins from the distinction:

```text
probable ≠ valid
confident ≠ calibrated
locally plausible ≠ globally coherent
prediction ≠ action-worthy inference
```

GSTL is an attempt to build a mathematical theory around this distinction.

---

## 2. The Core Idea From First Principles

Suppose a model proposes a candidate object `z`.

That object could be:

- a prediction,
- an output sequence,
- a classification,
- a plan,
- a repair,
- a policy action,
- an intervention,
- a generated image,
- a trajectory,
- a scientific hypothesis.

Classical learning usually assigns a loss or likelihood to `z`.

GSTL assigns a **validity residual**.

A residual measures how much the candidate violates a relevant contract:

```text
semantic mismatch
logical inconsistency
distribution shift
unsupported inference
calibration failure
constraint violation
unsafe action
repair failure
causal mismatch
```

The total residual is:

```math
R_G(z) = \sum_j w_j r_j(z)
```

The corresponding validity score is:

```math
\mathrm{CV}_G(z) = \exp(-R_G(z))
```

So the system does not only ask:

```text
How likely is z?
```

It also asks:

```text
How valid is z?
How uncertain is that validity?
What should we do if validity is low?
```

That is the core move.

---

## 3. Statistical Learning of Validity

GSTL treats validity as something that can be learned statistically.

The model can learn:

- what valid outputs look like,
- what invalid outputs look like,
- which residuals matter,
- how residuals interact,
- when a critic is uncertain,
- when a generator is exploiting a critic,
- when a prediction is out of support,
- when repair is likely to succeed,
- when to query for more information.

This gives a learned validity model:

```math
\widehat{\mathrm{CV}}_G(z) \approx \Pr(v(z)=1 \mid z)
```

where `v(z)=1` means the candidate is valid.

GSTL therefore turns validity into a first-class statistical object.

The goal is not only:

```text
learn p(y | x)
```

but:

```text
learn p(valid | x, y)
learn residuals
learn support
learn uncertainty
learn when to act
```

---

## 4. Why Thermodynamics?

Thermodynamics enters because GSTL studies learning systems that must trade off competing pressures:

- fit the data,
- reduce invalidity,
- preserve uncertainty when needed,
- avoid overconfident collapse,
- spend limited compute,
- choose safe actions,
- repair or fallback under failure.

This naturally leads to an energy-like view.

A candidate has likelihood energy, validity energy, uncertainty energy, and action cost:

```math
E_G(z)
=
E_{\mathrm{likelihood}}(z)
+
\beta R_G(z)
+
\gamma U_G(z)
+
\eta C(z)
```

GSTL then studies a free-energy style objective:

```math
\mathcal F_G(q)
=
\mathbb E_{z \sim q}[E_G(z)]
-
T H(q)
```

This says:

> Learn distributions that prefer low-energy, high-validity candidates while preserving enough entropy to explore, remain calibrated, and avoid premature collapse.

In plain language:

```text
Likelihood pulls the model toward what is common.
Validity pulls the model toward what is coherent.
Uncertainty prevents false confidence.
Thermodynamics describes the tradeoff.
```

---

## 5. The GSTL Closed Loop

A GSTL system is not just a trained model. It is a loop.

```text
1. Generate candidates
2. Score likelihood
3. Estimate validity
4. Estimate uncertainty
5. Rank or select
6. Act, repair, query, fallback, or stop
7. Store residual memory
8. Audit failures
9. Update the model
```

This makes GSTL a theory of **validity-governed learning systems**.

The mature loop is:

```text
prediction → validity → uncertainty → action → feedback → repair → redeployment
```

That loop is the heart of the framework.

---

## 6. Full-Sequence Ranking

For language models and sequence models, GSTL does not only score individual tokens.

It scores full sequences.

A standard autoregressive model assigns:

```math
\log q_\theta(y \mid x)
=
\sum_t \log q_\theta(y_t \mid x,y_{<t})
```

GSTL ranks complete candidate sequences by:

```math
J_G(x,y)
=
\log q_\theta(y \mid x)
-
\beta R_G(x,y)
```

or, using calibrated validity:

```math
J_G(x,y)
=
\log q_\theta(y \mid x)
+
\lambda_{\mathrm{CV}}
\log \widehat{\mathrm{CV}}_G(x,y)
```

So GSTL can choose a lower-likelihood sequence if it is much more valid.

This is essential because the most likely output is often not the best output.

---

## 7. Jump Inference

GSTL also supports **jump inference**.

If a model is generating step by step and enters a low-validity prefix, GSTL can jump to a better branch.

A standard model chooses:

```math
y_t^{\mathrm{MLE}}
=
\arg\max_a \log q_\theta(a \mid x,y_{<t})
```

GSTL chooses:

```math
y_t^{\mathrm{GSTL}}
=
\arg\max_a
\left[
\log q_\theta(a \mid x,y_{<t})
-
\beta R_G(x,y_{<t},a)
\right]
```

So if the most likely next token is invalid, GSTL can choose a different token because validity changes the control objective.

This turns validity into an active inference principle.

---

## 8. Repair, Query, Fallback, Stop

When validity is low, GSTL does not have to blindly output anyway.

It can choose among actions:

```text
accept
reject
repair
query
fallback
defer
stop
```

This is crucial for reliable AI.

A GSTL system should know not only what it predicts, but what to do when the prediction is not valid enough.

That makes GSTL naturally connected to:

- AI safety,
- uncertainty-aware deployment,
- tool use,
- agent reliability,
- self-repair,
- human-in-the-loop systems,
- scientific reasoning,
- formal verification,
- robust decision-making.

---

## 9. Validity Can Be Goodharted

If a generator optimizes a learned validity critic, it can exploit the critic.

This creates a Goodhart gap:

```math
\Delta_G
=
\mathbb E[\widehat{\mathrm{CV}}_G(z)]
-
\mathbb E[v(z)]
```

A high positive gap means the system thinks outputs are valid, but they are not.

GSTL therefore includes:

- adversarial audit,
- red-team candidate discovery,
- support diagnostics,
- multi-critic disagreement,
- calibration checks,
- fallback policies,
- deployment firewalls.

The lesson is simple:

> A learned validity critic must itself be audited.

---

## 10. Ensembles, Distillation, and Cascades

Full validity checking can be expensive.

GSTL therefore supports practical deployment strategies.

### Multi-Critic Validity

Use multiple critics:

```math
\mathcal E_G(z)=\{c_1(z),\ldots,c_M(z)\}
```

Mean validity:

```math
\mu_E(z)=\frac1M\sum_m c_m(z)
```

Disagreement:

```math
\sigma_E(z)
=
\left(
\frac1M\sum_m(c_m(z)-\mu_E(z))^2
\right)^{1/2}
```

Certification requires:

```math
\mu_E(z)\ge \tau_\mu,
\quad
\sigma_E(z)\le \tau_\sigma,
\quad
\min_m c_m(z)\ge \tau_{\min}
```

### Distillation

A cheap student critic can approximate the ensemble:

```math
d_\eta(z)\approx \mu_E(z)
```

But it should fallback to the ensemble when uncertain or out of support.

### Cascaded Validation

Easy cases use cheap validators. Hard cases escalate.

```text
cheap critic → medium critic → full ensemble → audit / fallback
```

The cascade processes only unresolved candidates:

```math
B_{k+1}
=
\{z\in B_k : \mathrm{Cert}_k(z)=0\}
```

This makes GSTL scalable.

---

## 11. What Makes GSTL Different?

GSTL is not just:

- a loss function,
- an energy model,
- a verifier,
- a calibration layer,
- an agent wrapper,
- a reranker.

It is a framework for learning systems in which validity governs the entire lifecycle:

```text
training
generation
selection
calibration
repair
audit
deployment
feedback
```

The core shift is:

```text
from predicting outputs
to governing valid inference.
```

---

## 12. Why This Matters for AI

As AI systems become more autonomous, the critical question becomes less:

> Can the model produce an answer?

and more:

> Should this answer be trusted, acted on, repaired, queried, or rejected?

GSTL is designed for that world.

It gives a language for systems that must reason about:

- validity,
- uncertainty,
- support,
- consistency,
- repairability,
- deployment risk,
- action consequences.

This is why GSTL is naturally relevant to:

- language models,
- scientific discovery,
- planning systems,
- autonomous agents,
- theorem proving,
- forecasting,
- robotics,
- medicine,
- finance,
- safety-critical AI.

---

## 13. Minimal Mathematical Core

The protected GSTL core is:

```math
G_{\mathrm{core}}=(L,M,\mathsf P,I)
```

where:

- `L` is the statistical learning object,
- `M` is the learned relevance geometry or metric,
- `P` is the posterior and uncertainty protocol,
- `I` is the validity contract system.

The operational GSTL extension is:

```math
G_{\mathrm{GSTL}}
=
(L,M,\mathsf P,I,\mathsf R,\mathsf K,\mathsf A,\mathcal M,\mathcal C,\mathcal D)
```

where:

- `R` is the residual and validity system,
- `K` is calibration,
- `A` is the action policy,
- `M` is residual memory,
- `C` is candidate generation,
- `D` is deployment governance.

The central object is not a prediction alone.

The central object is:

```text
a validity-conditioned learning-control loop.
```

---

## 14. The One-Sentence Vision

> **GSTL turns validity from a diagnostic into a control principle for learning systems.**

Or even shorter:

> **Learning should know when it is valid.**

---

## 15. Manual

This README is the executive summary.

The full mathematical manual develops the framework in detail, including definitions, derivations, theorem sketches, diagnostics, and extensions.

Recommended reading:

```text
GSTL_Core_Framework_Theory_Manual_v266_FINAL.pdf
```

---

## 16. Current Research Direction

The current research frontier is moving toward GSTL transformers:

```text
classic transformer
+
implicit validity learning
+
sequence-space ranking
+
jump inference
+
calibrated validity
+
residual memory
+
deployment cascade
```

The key experimental question is:

> Can GSTL models learn better true predictions than maximum-likelihood models when validity reveals structure hidden by biased data?

That is the next major test.

---

## 17. Final Manifesto

A learning system should know not only what it predicts.

It should know:

```text
when its inference is valid,
when its confidence is calibrated,
when its context is out of support,
when it should repair,
when it should ask,
when it should fallback,
when it should stop,
and when it is safe to act.
```

That is the purpose of GSTL.
