# GSTL — Generalized Statistical Thermodynamic Learning

**GSTL is a framework for learning systems that do more than predict.**

Most machine learning asks:

> What is the most likely output?

GSTL asks:

> Is this output valid enough to trust, act on, repair, or reject?

The core idea is simple:

```text
prediction → validity → uncertainty → action
```

GSTL treats **validity** as a first-class object of learning. A model should not only estimate likelihood; it should also learn when its inference is coherent, supported, calibrated, repairable, and safe to use.

## Why GSTL?

Modern AI systems can produce fluent, high-confidence outputs that are still wrong, unsupported, inconsistent, or unsafe.

GSTL is motivated by the gap between:

```text
probable output
```

and:

```text
valid inference
```

It combines ideas from:

- statistical learning
- thermodynamics and free energy
- uncertainty and calibration
- residual/error modeling
- validity contracts
- repair, fallback, and audit loops
- sequence ranking and jump inference

The goal is to build learning systems that can say:

```text
I can answer.
I am uncertain.
This output violates a contract.
I should repair it.
I should ask for more information.
I should fallback.
I should not act.
```

## Core intuition

GSTL assigns candidates a residual validity energy:

```math
R_G(z)=\sum_j w_j r_j(z)
```

and a corresponding validity score:

```math
\mathrm{CV}_G(z)=\exp(-R_G(z)).
```

A GSTL system therefore ranks candidates not only by likelihood, but by likelihood plus validity:

```math
J_G(z)=\log q_\theta(z)-\beta R_G(z).
```

So the best output is not merely the most probable output.

It is the output that best balances:

```text
likelihood
validity
uncertainty
cost
actionability
```

## What GSTL is trying to become

GSTL is being developed as a general theory for **validity-governed learning systems**.

The long-term direction is:

```text
predictive models
→ validity-aware models
→ self-auditing models
→ repair-capable models
→ uncertainty-calibrated agents
→ thermodynamic learning-control systems
```

Current research is exploring GSTL in the context of:

- transformers
- sequence-to-sequence learning
- implicit validity learning
- candidate reranking
- jump inference
- calibrated validity critics
- adversarial audit
- cascading validation systems

## Full theory manual

This README is only a short overview.

For the full mathematical framework, definitions, derivations, theorem sketches, diagnostics, and research roadmap, read the PDF manual in this repository:

```text
GSTL_Core_Framework_Theory_Manual_v266_FINAL.pdf
```

## One-sentence vision

> **GSTL turns validity from a diagnostic into a control principle for learning systems.**
