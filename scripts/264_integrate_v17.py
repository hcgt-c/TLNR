#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""264_integrate_v17.py — build paper/paper_v17.md from paper/paper_v15.md.

v17 = v15's organisation + v16's version-independent corrections (the decision recorded in
paper/V17_INTEGRATION_PLAN.md).  The script authors no number: every replacement below is text, and
every literal it inserts is either copied from paper_v16.md or written in V17_INTEGRATION_PLAN.md.
Each operation asserts that its markers occur exactly once, so the build fails loudly if v15 moves.

Usage:  python scripts/264_integrate_v17.py
Output: paper/paper_v17.md, results/v17_integration.json
"""
import json
import os
import re
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(WORK, 'paper', 'archive', 'paper_v15.md')
DST = os.path.join(WORK, 'paper', 'paper_v17.md')

OPS = []


def span(name, start, end, new):
    OPS.append(('span', name, start, end, new))


def lit(name, old, new):
    OPS.append(('lit', name, old, None, new))


# ---------------------------------------------------------------- abstract / §1
span('abstract',
     'How neural representations preserve the structure of input changes connects',
     'connecting the understanding of neural representations to their design.',
     r"""How neural representations preserve the structure of input changes connects representation analysis with internal intervention. We study operable representational content through compatible actions of reference transformations on neural features. We characterise when a transformation descends through an encoder, and give a linear setting in which the defect is governed by the transformation's demand for discarded information, measured in the metric the representation induces. On a rectifier the failure to realise a transformation has two distinguishable sources — what the source region has already made unrecoverable, and what it costs to satisfy every region the transformation visits with one operator — and for a *measured* harmonic carrier the same question has a closed answer: a linear realisation exists exactly when the retained harmonic blocks are invariant under the action. Using colour as the in-depth instance, we find that hue orbits in frozen visual features concentrate 84–88% of their energy in the first two harmonics with rotation planes shared across shapes, that this organisation is substantially inherited from input and architecture and is reshaped by training and depth, and that the measured structure supports prediction, transport from new starting states, and composition — with global and local realisations differing sharply in which they achieve. Guided by the measurements, we construct a compact interface whose rotation action is fixed by the structure and never fitted: it reads hue zero-shot at 3.4° median error on unseen shapes. Theory, structural measurement, and construction together establish transformation laws as a concrete object connecting the understanding of neural representations to their design.""")

span('sec_1_2_results',
     '**Existence is an information condition, not a score.**',
     'a layer\'s operability is a property of the pair. *Earned at* §3.5, §6.1–6.3.',
     r"""**Existence is an information condition, and its failure is priced twice over.** An induced action exists at a site exactly when the transformation preserves that site's fibres: states the encoder has merged must stay merged after the transformation. Existence is therefore a property of the encoder–transformation pair, and it is a single condition rather than two, because once the action is well defined its composition law is inherited from the physical one. In the linear layer the same condition has an explicit form — a realising operator exists iff the transformation preserves the encoder's kernel — and its four-block decomposition shows that *only* the block that routes discarded information back into the retained subspace obstructs: discarding information is free, contradicting it is not. When a realising operator exists it is unique; when it does not, the best achievable score is set by the mixing ratio *in the metric the representation induces*, $1-T_{RMS}=\kappa_W/\sqrt{1+\kappa_W^2}$ under stated moment conditions, verified to $0.0091$ over a $\kappa$ sweep spanning sixteen orders of magnitude. Two consequences are counter-intuitive and both are measured: only mixing-back is poisonous, and retained rank is nearly irrelevant — $\rho(\kappa,T_F)=-0.968$ against $\rho(\text{effective rank},T_F)=+0.502$ on trained weights. The law's tightness condition is explicit, and real backbones do not meet it; the paper reports the deviation rather than presenting a controlled-regime identity as a calibrated predictor. On a rectifier the failure to realise has two distinguishable sources: what the source region has already made unrecoverable on a transition cell, and what it costs to satisfy every cell the transformation visits with one operator. *Earned at* §3.2–3.5, §8.3.

**The measured organisation fixes which realisations can exist.** A continuous rotation action on features is block-diagonal in a suitable basis, so the measurement question is which blocks carry the energy and whether their planes are shared. For colour, the first two harmonics carry $84$–$88\%$ of the orbit energy and the planes are shared across shapes at $\cos\ge0.96$, and that form is prescriptive: for a carrier $B$ with action $A_\Delta$, a fixed linear realisation exists iff $A_\Delta\ker B\subseteq\ker B$, and then the least-squares optimal map is $K^\star=BA_\Delta B^\dagger$, whose residual is the energy of the retained-block directions the rotation carries back into view. The criterion is a *design* condition for a compiled carrier — it says which coordinate sets may be kept and what a chosen truncation costs. On the measured features the synthesis is injective, so closure holds exactly and the residual failure of a global fit is attributable to content dependence and tail instead; the paper keeps that distinction rather than blurring it. *Earned at* §4.2–4.6.

**From the correspondence to operation: what it costs, and what it builds.** Global linear families fail where a local, phase-conditioned transporter succeeds; chains compose within a bound set by the truth action's Lipschitz constant in the consumer's seminorm, at a price that is measured rather than assumed; and the measured form fixes an interface whose action is never fitted, whose error has three named sources of which only one is removed by training, and whose operating domain is published as a threshold rule. *Earned at* §5, §7.""")

span('contrib_1',
     '1. **Theory: when a transformation law is realisable at all**',
     'a term linear in the plane misalignment (Theorem 5).',
     r"""1. **Theory: when a transformation law is realisable at all** (§3, §4.6, §5.2, §5.4, §7.3). A realisability theory in three layers. *General*: an induced action exists exactly when the transformation preserves the encoder's fibres, and existence already yields the homomorphism (§3.2: the universal property of the quotient, with its two consequences). *Single-region*: a linear operator exists iff the kernel is preserved, and its defect obeys the metric-weighted closed form of §1.2 (Theorem 1, Theorem 2). *Piecewise-linear*: on a rectifier encoder the effect decomposes exactly into a within-regime term and a term carried by the pairs of regions the transformation moves between; whether one fixed linear operator can satisfy every visited pair is a decidable joint system (Propositions C, D), and for the fixed-linear class achievable faithfulness is bounded by the consumer-visible share of the change (Proposition E). Four further statements cover realisation and composition: a carrier admits a fixed linear action exactly when its retained blocks are closed, in which case the least-squares optimal map is the carrier action seen through the truncation (Propositions F–G); a closed orbit has no exact coordinate-monotone realisation (Theorem 3); chain error obeys a three-regime accumulation law set by the Lipschitz constant of the truth action in the consumer's seminorm (Theorem 4); and a carrier with band-limited orbits and shared rotation planes realises the fixed block-diagonal action with error bounded by the read-in error, the action mismatch and the target tail (Theorem 5).""")

lit('glance_mixing_row',
    r'| Realisation is governed by the mixing ratio; only mixing-back is poisonous; retained rank nearly irrelevant | $\kappa$ sweep $4.5\times10^{-16}\to9.34$, closed form within $0.0091$; $\rho(\kappa,T_F)=-0.968$ against $\rho(\text{rank})=+0.502$ | linear encoders, stated moments |',
    r'| Realisation is governed by the mixing ratio in the induced metric; only mixing-back is poisonous; retained rank nearly irrelevant | $\kappa$ sweep $4.5\times10^{-16}\to9.34$, closed form within $0.0091$; $\rho(\kappa,T_F)=-0.968$ against $\rho(\text{rank})=+0.502$ | linear encoders, stated moments |')

lit('glance_perdim_row',
    r'| Per-dimension families are provably insufficient; local transport succeeds where global fails | third principal value $\approx10\%$ of orbit energy; $10\times$ tighter, loop closure $0.4\%$ | colour sites |',
    r'| A carrier is usable iff its retained blocks are closed; then the optimal fixed linear map is the projected carrier action | closure criterion verified on 40 random selections; $K^\star=PAP^{\mathsf T}$ to $9\times10^{-16}$; $10\times$ tighter local transport | colour sites |')

# ---------------------------------------------------------------- §2 related work
lit('sec_2_3_ref',
    'only when we ask a representation to carry it (§4.6).',
    'only when we ask a representation to carry it (§4.7).')

lit('sec_2_4_koopman',
    'Koopman-style approaches lift temporal dynamics into an approximately linear embedding, of which our band-limited global family is a truncated instance applied to physical transformations.',
    r"""Koopman-style approaches ask when a nonlinear system admits an invariant finite-dimensional subspace on which the evolution is linear, and design or learn a lift into it; our band-limited global family is a truncated instance of that lift applied to a *physical* transformation of a frozen vision encoder. What that line does not supply — and what this paper adds — is an existence criterion for a *given* representation rather than a designer's lift, together with the transfer from temporal dynamics to non-temporal transformations of images.""")

# ---------------------------------------------------------------- §3 theory
lit('map_T2_row',
    "| T2 | The defect law 1 − T_RMS = κ/√(1+κ²) | error = the C-block's energy against the demanded displacement | §3.4, Thm 2 (proof + sweep) |",
    r"| T2 | The defect law $1-T_{RMS}=\kappa_W/\sqrt{1+\kappa_W^2}$ in the induced metric | error = the $C$-block's energy against the demanded displacement, weighted by the encoder | §3.4, Thm 2 (proof + sweep) |")

lit('map_T5_row',
    r'| T5 | The deviation of the linear law at trained sites **is** the crossing term | Prop D decomposes the effect; the crossing term is a function of the region index and carries the deviation | §3.5, §6.5 (proof + 42 sites) |',
    r'| T5 | The deviation of the linear law at trained sites is a change of local rule | Prop D decomposes the effect; the deviation is carried by the diversity of the realised rules | §3.5, §6.5 (proof + 42 sites) |')

lit('map_T6_row',
    r'| T6 | Depth erosion follows the readout–transformation pairing | the crossing share grows with depth (Prop D), and a crossing counts only where the consumer reads — the two statements are one | §3.5, §6.3 (proof + 2×2 + visible share) |',
    r'| T6 | Depth erosion follows the pairing between the transformation and the readout | rule diversity is created by gate changes (Prop D), and a change of rule counts only where the consumer reads it | §3.5, §6.3 (proof + 2×2 + visible share) |')

lit('map_T7_row',
    r'| T7 | Achievable faithfulness is bounded by the consumer-visible crossing share | Prop E, for the fixed-linear class | §3.5 (proof sketch + 6/6 empirical) |',
    r"""| T7 | Achievable faithfulness is bounded by the consumer-visible share of the change | Prop E, for the fixed-linear class | §3.5 (proof sketch + 6/6 empirical) |
| T8 | A carrier admits a fixed action iff its retained blocks are closed, and then the optimal linear map is the projected carrier action | Prop F, Prop G | §4.6 (proof + 40 random selections) |""")

lit('map_closing',
    'T1–T4 and T7 are proved or proved-then-verified; T5–T6 are consequences of the piecewise-linear layer of §3.5 read on real networks.',
    'T1–T4 and T7–T8 are proved or proved-then-verified; T5–T6 are consequences of the piecewise-linear layer of §3.5 read on real networks.')

lit('sec_3_3_heading',
    '### 3.3 Quantifying irrecoverable and realisation error',
    '### 3.3 Two sources of realisation error')

span('sec_3_3_append',
     '### 3.4 A quantitative linear model',
     '### 3.4 A quantitative linear model',
     r"""The same two-source structure appears concretely on a rectifier, and there it is measurable. A rectifier network partitions its input space into *activation regions* $R_r$ indexed by the gate pattern, and on each region the site map is affine, $\phi(x)=A_rx+b_r$; the gate is recoverable from the site tensor itself, since $z=\mathrm{ReLU}(a)$ gives $G=\mathbb 1[z>0]$. Write $u=\phi(x)=A_rx+b_r$ for the source feature and $v=\phi(Tx)=A_sTx+b_s$ for its image under a linear input transformation $T$, and index by *transition cell* $(r,s)$ the inputs that start in region $r$ and land in region $s$. The realisation error of a fixed linear map decomposes into two terms with different meanings,
$$\min_K\ \mathbb E\|Ku-v\|^2=\underbrace{\sum_{r,s}w_{rs}\min_{K_{rs}}\mathbb E\big[\|K_{rs}u-v\|^2\,\big|\,(r,s)\big]}_{\text{within-cell error}}+\underbrace{\sum_{r,s}w_{rs}\,\mathbb E\big[\|(K_{rs}-K^\star)u\|^2\,\big|\,(r,s)\big]}_{\text{cost of sharing one operator}},$$
with $w_{rs}$ the cell weights, $K_{rs}$ the cell-wise least-squares operators and $K^\star$ the pooled one; the split is the orthogonal decomposition of a pooled regression into within-cell and between-cell parts.

*The first term need not vanish*, and its source is broader than information loss. On a cell $u$ and $v$ are both affine in $x$, which does not make $v$ affine in $u$: if the source rule has already discarded what the target rule needs — one region covering everything, with $u=x_1$ and $v=x_2$ — no choice of operator helps, and that is irrecoverability localised to a cell. But a positive within-cell term can also come from the restricted form of the family (a zero-intercept linear fit where the cell relation has an offset, say), so the within-cell term is the error of the *best member of the family on that cell*, not a measure of discarded information. When the source rule does determine the target rule on the cell and the family can express it, the first term vanishes and the whole error is the cost of sharing.

*The second term is what a change of gate pattern can create.* A fixed linear $K$ realises the transformation on the visited cells iff the joint system of §3.5 is consistent; since that system is linear in $K$, consistency is a rank condition, and its least-squares residual is the data-weighted sharing cost when the cell feature moments are matched — the general statement is the moment-weighted version of the decomposition above. Two qualifications belong to it: the cells must cover the region with non-empty interior, or span it, for a zero finite-sample residual to imply the matrix identity on the region; and the criterion concerns the *visited* cells, which is what a measurement can see.

### 3.4 A quantitative linear model""")

span('thm2_main',
     '**Theorem 2 (defect law).** Under the stated moment conditions (Appendix B)',
     '**Table 2** carries the controlled verification and the scope of the law.',
     r"""**Theorem 2 (defect law).** Under Assumptions B and C the best score is governed by the mixing ratio *in the metric the representation actually induces*: with $A$ and $C$ the retained blocks and $W$ the encoder, the least-squares optimum satisfies
$$1-T_{RMS}=\frac{\lVert WC\rVert_F}{\sqrt{\lVert W(A-I_{\mathcal R})\rVert_F^2+\lVert WC\rVert_F^2}}\;=\;\frac{\kappa_W}{\sqrt{1+\kappa_W^2}},\qquad \kappa_W:=\frac{\lVert WC\rVert_F}{\lVert W(A-I_{\mathcal R})\rVert_F}.$$
When $W$ is an isometry on the retained subspace — in particular on whitened features — this is $\kappa=\lVert C\rVert_F/\lVert A-I\rVert_F$ with the optimum at $A$; in general the optimum is $W$-conjugate. *Proof.* The residual is $WCx_K$ and the displacement is $W(A-I)x_R+WCx_K$; under isotropy $x_R\perp x_K$, so $\mathbb E\lVert WCx_K\rVert^2=\lVert WC\rVert_F^2$ and $\mathbb E\lVert W(A-I)x_R\rVert^2=\lVert W(A-I)\rVert_F^2$, and dividing through gives the displayed form. $\square$

Two readings are counter-intuitive, and the first is metric-independent: **only mixing-back is poisonous** — the $D$ block loses information harmlessly while $C$ contradicts it, so *discarding information is not what hurts* — and **retained rank barely matters**, since the split enters only through $\kappa_W$. The metric sets the magnitude rather than the mechanism, which is why a law calibrated at one site need not transfer to another with a different $W$. **Table 2** carries the controlled verification and the scope of the law.""")

span('scope_of_law',
     '**Scope of the law.** The closed form is exact under isotropy (Assumption C)',
     'and the diagnostic use of the deviation.',
     r"""**Scope of the law.** The closed form is exact under isotropy (Assumption C), and its tightness condition is stated in Proposition B.3: under a general covariance the identity holds only when a cross-covariance term vanishes, and Corollary B.4 shows the score then depends on the input covariance. Real sites are strongly anisotropic, so the law is a statement about the controlled regime rather than a calibrated predictor for real backbones. On the 42 reachable sites of the depth grid, with three estimators of the retained subspace and five ranks and no constant fitted to the score, the prediction is biased high by $+0.40$ to $+0.65$ and orders no better than the naive baselines — a metric mismatch, since each site brings its own $W$ and its own covariance. What transfers is the mechanism (alignment rather than retained rank) and the diagnostic reading of the deviation; the magnitude is a per-site quantity.""")

span('propC_D_main',
     '**Proposition C (multi-region compatibility).** Suppose $\\tau$ leaves the gate pattern of the visited regions unchanged',
     'significance levels and source files in Appendix C.7).',
     r"""**Proposition C (region-transition criterion).** Let the site be affine on each visited region, $\phi=A_rx+b_r$, let $T$ be the input transformation, and let $(r,s)$ range over the realised transitions — the cells on which $x$ starts in region $r$ and $Tx$ lands in region $s$, each with non-empty interior or spanning its region. Then a fixed linear $\rho$ with $\rho\phi=\phi\circ T$ on the visited cells exists **if and only if** the joint system
$$\rho A_r=A_sT,\qquad \rho b_r=b_s\qquad\text{for every realised }(r,s)$$
is consistent. *Proof.* On a cell, $\phi(Tx)=A_sTx+b_s$ and $\rho\phi(x)=\rho(A_rx+b_r)$; both sides are affine in $x$, and an identity between affine maps on a set with non-empty interior extends to the maps, so the system is necessary; conversely a solution satisfies it on every cell. $\square$

**Corollary C.1 (no crossing).** If $T$ leaves the gate pattern of each visited region unchanged, the realised transitions are the diagonal cells and the system reduces to $\rho A_r=A_rT$, $\rho b_r=b_r$ for every visited $r$. For a single visited region this is Theorem 1's kernel condition rewritten — the linear layer is the single-region case.

*Consequence.* **Even with no gate change at all, the multi-region structure alone can obstruct a fixed linear action**: the visited regions must be simultaneously conjugate to the transformation. This is why enlarging the operator family buys nothing — measured, a ridge-plus-$3\times3$-residual family and the global linear family agree to within $0.087$ at every site (§5.2).

**Proposition D (the decomposition, and where rule diversity comes from).** With $z=\mathrm{ReLU}(a)$, $a'=a+\Delta a$, $G=\mathbb 1[a>0]$ and $G'=\mathbb 1[a'>0]$,
$$z'-z=G\odot(a'-a)+(G'-G)\odot a',$$
and on a unit whose gate changes the second term is $+a'$ (off$\to$on) or $-a'$ (on$\to$off). *Proof.* $z'=G'\odot a'=G\odot a'+(G'-G)\odot a'$ and $z=G\odot a$. $\square$

*Neither direction of implication between gate changes and failure holds.* Take $\phi(x)=(\mathrm{ReLU}\,x,\mathrm{ReLU}(-x))$ with $Tx=-x$: every nonzero input changes one gate, and the fixed operator that swaps the two coordinates realises the transformation **exactly**, since $\phi(-x)=K\phi(x)$ for all $x$. For the other direction take the scalar site $\phi(x)=\mathrm{ReLU}(x_1)+\mathrm{ReLU}(x_2)$ with $T=\mathrm{diag}(2,3)$, which *preserves every gate pattern*, since positive scalings do not change signs. On the region where the first unit is active, $u$ and $v$ are related by one factor; where the second is active, by the other. Each region is exactly realisable on its own, yet a single scalar $K$ would have to equal both factors at once. No gate change occurs and no shared operator exists. Gate changes are therefore the *source* of rule diversity, not the obstruction; the obstruction is whether one operator satisfies every rule the transformation actually visits, which is the joint system of Proposition C.

*Consequence (which quantity governs).* The within-regime term is linear in $\Delta a$ and is what a linear-class action can reach. The gate-change term has magnitude $|a'|$ on the changed units, so it is not a constant that vanishes with the perturbation, and no fixed linear operator represents a function of the discrete region index. What governs the achievable score is therefore the *concentration* of the change and its visibility to the consumer, not the number of gate flips. Measured (ResNet-50, three depths × two algebras): the flip count does not order the score ($\rho=+0.49$, not significant), the crossing share does ($\rho=-0.83$), and the consumer-visible share orders it most tightly of all ($\rho=-0.886$; significance levels and source files in Appendix C.7; the measured associations are proxies for the sharing cost of Proposition C, not the cost itself).""")

lit('sec_3_5_closing',
    "This is the layer in which the paper's real-network results live. The deviation of the linear law at trained sites (§6.5) is its crossing term — a quantity the framework predicts, not an anomaly — and the depth trend (§6.1) is the statement that the crossing share grows with depth.",
    "This is the layer in which the paper's real-network results live. The deviation of the linear law at trained sites (§6.5) is a change of local rule — a quantity the framework predicts, not an anomaly — and the depth trend (§6.1) is the statement that the transformation visits more distinct rules as depth grows, which §6.6 measures.")

lit('claim_scope_thm2_row',
    r'| Kernel-preservation criterion, κ defect law | **Single-region (linear) encoders**, stated moment conditions | §3.4 (proofs + controlled verification) |',
    r'| Kernel-preservation criterion, metric-weighted defect law | **Single-region (linear) encoders**, stated moment conditions | §3.4 (proofs + controlled verification) |')

lit('claim_scope_prop_row',
    r'| Multi-region compatibility (Prop C), crossing decomposition (Prop D), achievable-score bound (Prop E) | **Piecewise-linear encoders**; E for the fixed-linear class, measured | §3.5 (proofs; E empirical) |',
    r'| Region-transition criterion (Prop C, Cor. C.1), gate-change decomposition (Prop D), achievable-score bound (Prop E) | **Piecewise-linear encoders**; E for the fixed-linear class, measured | §3.5 (proofs; E empirical) |')

lit('claim_scope_thm3_row',
    r'| Capacity boundary of per-dimension families (Thm 3) | Fixed-basis per-dimension monotone families, orbit with σ₃² > 0 | §5.2 (proof + spectrum) |',
    r'| No exact coordinate-monotone realisation of a closed orbit (Thm 3) | Monotone parameter families in a fixed basis | §5.2 (proof + measured best fit) |')

lit('claim_scope_carrier_row',
    r'| Carrier sufficiency (Thm 5) and the interface | Any site meeting the measured premises; instantiated for colour | §7.3 (proof + measured premises) |',
    r"""| Carrier closure and optimal linear realisation (Props F–G) | Any carrier and action; a **design** condition on the retained coordinates | §4.6 (proofs + 40 selections) |
| Carrier sufficiency (Thm 5) and the interface | Any site meeting the measured premises; instantiated for colour | §7.3 (proof + measured premises) |""")

# ---------------------------------------------------------------- §4 closure design condition
span('sec_4_6_insert',
     '### 4.6 Beyond hue: the protocol\'s reach',
     '### 4.6 Beyond hue: the protocol\'s reach',
     r"""### 4.6 When the measured carrier supports a linear action: a design condition

The measurements above describe the *form* of the orbit. What that form licenses has an exact answer, and it is a condition on the carrier one chooses to keep. Summarise the measured organisation by a carrier map $B$ from $m$ carrier coordinates $c$ (the harmonic blocks) into the feature space, $z=Bc$, and suppose the input transformation acts on the carrier by a known $A_\Delta$ — for hue, the block rotations $R(k\Delta)$ on the $k$-th pair.

**Proposition F (closure criterion).** There exists a fixed linear $K$ with $KB=BA_\Delta$ if and only if $A_\Delta\ker B\subseteq\ker B$. *Proof.* $KB=BA_\Delta$ is solvable iff $BA_\Delta$ vanishes on $\ker B$. $\square$

Closure is a property of the *retained blocks*, decidable from the measured organisation alone: $\ker B$ consists of the harmonic directions the carrier does not keep, and the criterion asks whether the rotation carries them back into the retained ones.

**Proposition G (optimal linear realisation).** Let $\mathbb E[cc^\top]=I$, $z=Bc$ and $z'=BA_\Delta c$. Then $K^\star=BA_\Delta B^\dagger$ is a least-squares minimiser and
$$\min_K\ \mathbb E\|Kz-z'\|^2=\big\|BA_\Delta P_{\ker B}\big\|_F^2,$$
which vanishes exactly when Proposition F holds. For a coloured carrier, $\mathbb E[cc^\top]=\Sigma_c$, the residual is $\|BA_\Delta\Sigma_c^{1/2}P_{\ker(B_\Delta\Sigma_c^{1/2})}\|_F^2$ with $K^\star=BA_\Delta\Sigma_cB^\top(B\Sigma_cB^\top)^{\dagger}$ and $\Sigma_c^{1/2}$ the **symmetric** square root. The projector must be taken *after* whitening: correlation lets a retained coordinate predict a discarded one, so the optimal predictor is not the Euclidean projection onto $\ker B$. The two expressions coincide only for $\Sigma_c\propto I$. *Proof.* $K^\star z-z'=BA_\Delta(B^\dagger B-I)c=-BA_\Delta P_{\ker B}c$, because $B^\dagger B$ is the orthogonal projector onto the row space of $B$; taking expectations with $\mathbb E[cc^\top]=I$ gives the Frobenius norm, and the coloured case follows by whitening. $\square$

Three consequences make this a design statement. *Existence:* the same expression that decides whether an action exists also prices its failure. *Error source:* what a fixed linear map on the carrier cannot do is exactly to undo the carrier directions the rotation carries back into view — the merged harmonics re-entering the retained subspace, the harmonic form of the mixing-back block of §3.4. *Design:* a carrier is usable iff its retained coordinate set is invariant under the action, which for a rotation with one two-dimensional block per frequency means keeping or dropping each conjugate pair **as a pair**. (The pair rule is stated for a full continuous rotation action written in one block per frequency, with selection along those coordinates; for a single fixed angle, or when frequencies repeat and their blocks may mix, the right object is the invariant subspace rather than the pair.)

Two numerical checks fix the scale, and the full columns are Table D.21. On the three-dimensional code $z(h)=(\cos h,\sin h,\cos 2h)$ — information-complete for the hue and entirely within the first two harmonics — the best fixed linear map leaves residual $0.96$ and closure fails, because $\cos2h$ alone is not invariant: the rotation sends it into the missing $\sin2h$. Adding that one coordinate makes the action numerically exact with $A_\Delta=\operatorname{diag}(R(\Delta),R(2\Delta))$, and $K^\star$ is then the block rotation itself. Over random coordinate selections of a six-dimensional carrier, the minimiser equals the projected carrier action $PA_\Delta P^\top$ to machine precision, and the closure verdict coincides with the sign of the residual in every case. *Source: `scripts/259_carrier_optimality_identity.py`; `results/carrier_optimality_identity.json`.*

**What the criterion does and does not explain on measured features.** The same analysis, run on features rather than on a synthetic code, is informative in a different way. Taking the carrier of the known input hue and estimating the synthesis $B$ from the features, the reconstruction tail is a few percent at two sites, the closure defect is numerically zero — because with a feature dimension far larger than five the synthesis is injective, so the exact carrier model is closed automatically — and a global linear map fitted on half the hues nevertheless leaves a large relative error on the other half (Table D.21). On these sites the binding constraint is therefore neither closure nor the tail but *cross-content inconsistency*: the carrier-to-feature map is content-dependent, which is the same fact as the shape-specific envelopes of §4.3 and the local-versus-global gap of §5.3. Dropping the $\sin2h$ coordinate raises the tail and leaves the closure defect at zero for an injective synthesis, as the pair rule predicts. *Source: `scripts/261_real_site_closure.py`; `results/real_site_closure.json` (dense5 orbits at the YOLO11n $y3$ and ResNet-18 $l2$ sites, $37°$ shift, half the hues fitted and the other half held out).*

That reading and the content-conditioning result of §5.2 are compatible only if their settings are kept apart, and the paper states both rather than renaming one. Content conditioning did not rescue global forms in 48 matched-capacity contrasts, including an oracle partition by the true displacement direction (Table 3); that experiment conditions a *fitted operator* on content, at matched capacity, on the arms it names. The cross-content statement above is about a *linear map on a measured carrier* whose synthesis is estimated per site, and it names content dependence as what remains after closure and tail are removed. The two differ in model, data, conditioning information and capacity, and one conclusion follows from them jointly: a content-conditioned family that fails while *using* content information is evidence that the failure is not cured by adding content *capacity*. What the criterion is for, in this paper, is narrower and unambiguous — it says which coordinate sets a compiled carrier may keep, and what a chosen truncation costs.

The measured organisation therefore has operational content in a precise sense. Harmonic concentration (§4.2) says a *small* carrier suffices; the shared planes (§4.3) say one $A_\Delta$ serves every shape; and Proposition F says which coordinates must be kept for the action to exist at all. §5 starts from that carrier, and §7 builds the interface on it.

### 4.7 Beyond hue: the protocol's reach""")

lit('sec_4_7_ref_a',
    'described in §4.6 is what a measurement would require.',
    'described in §4.7 is what a measurement would require.')

lit('sec_4_7_ref_b',
    'perimeter row of §4.6 and this project\'s supplement',
    'perimeter row of §4.7 and this project\'s supplement')

# ---------------------------------------------------------------- §5 realisation
lit('sec_5_2_content_conditioning',
    'Content conditioning does not rescue global forms (48 matched-capacity contrasts, including an oracle partition by the true displacement direction).',
    'Content conditioning does not rescue global forms (48 matched-capacity contrasts, including an oracle partition by the true displacement direction); §4.6 states what that negative result does and does not imply about the cross-content reading of the residual error at measured sites.')

lit('sec_5_2_thm3_intro',
    'One of the rejected families fails **by theorem rather than by observation**, and the theorem is quantitative: it fixes the failure scale of the whole family class before any fitting begins.',
    'One of the rejected families fails **by theorem rather than by observation**.')

span('thm3_main',
     '**Theorem 3 (capacity boundary of per-dimension families).** If the orbit',
     r'All three are written out in Appendix B (Lemmas B.7, B.8). $\square$',
     r"""**Theorem 3 (a closed orbit has no exact coordinate-monotone realisation).** A family $\{G_t\}$ whose members are coordinate-wise monotone in a fixed basis and whose dependence on the family parameter is monotone — $G_t(z)_i=f_{i,t}(z_i)$ with $(t,u)\mapsto f_{i,t}(u)$ non-decreasing in both arguments — realises an orbit only if every coordinate of that orbit is monotone in the parameter. Hence a closed non-degenerate orbit admits no exact realisation by such a family, and each coordinate that decreases by $A$ over intervals of length $\ell$ contributes $L^2$ error at least $A^2\ell/4$.""")

span('thm3_followup',
     'The measured spectrum closes the argument numerically:',
     'is the escape the theorem permits.',
     r"""The theorem says what the class cannot do; the measurement says what it costs. On the reference orbit the best coordinate-monotone fit carries $L^2$ error $998$ against $924$ for the best fixed linear map (Table 3), and both fail by a wide margin, while the carrier family that keeps the conjugate pair is exact. Segmenting the basis — letting it follow the field locally, §5.3 — is the escape the theorem permits and the measurements confirm.""")

lit('table3_monotone_row',
    r'| monotone per-dimension | bounded by $\sigma_3^2$ (Thm 3) | fails by theorem | rejected on capacity grounds |',
    r'| monotone per-dimension | no exact realisation of a closed orbit (Thm 3); best fit measured, $L^2=998$ on the reference orbit | fails by theorem and by measurement | rejected |')

span('thm4_discussion',
     'The bound separates three regimes by one property of the truth:',
     'the two regimes measured are the contractive and the isometric ones.',
     r"""The bound separates three regimes by one property of the truth: $L<1$ (contractive) gives a **bounded** geometric accumulation $\varepsilon/(1-L)$; $L=1$ (isometric) gives **linear** accumulation $n\varepsilon$; $L>1$ (expansive) gives **exponential** growth $L^n$. The two tested families differ in how their chains behave, and the accumulation law is the tool that separates the possible causes: a contraction caps the chain error, an isometry accumulates it linearly, an expansion grows it exponentially. Which regime a site occupies must be *measured* in the consumer's seminorm, not inferred from the generator. Two readings follow, and they are worth separating. Heat's monotone degradation in Table 5 is therefore **not** amplification by the truth action — a contraction forbids that — but a growing single-step error as the chain wanders into states where the fitted family is worse; and hue's flat chains, under the loosest of the three bounds, mean that its single-step error is already small rather than that the bound protects it. The regime that would actually blow up, $L>1$, is not instantiated in this paper. In all three cases the composition law itself is automatic; what the accumulation law prices is the realisation error. Estimating $\varepsilon$ and $L$ independently of the measured chain error, and instantiating the expansive class ($L>1$), where the bound predicts exponential growth, are not done here.

**Metric distortion at real sites.** The one number Theorem 4 needs is how the truth action distorts distances in the consumer's seminorm, and it is measurable. Over pairs of held-out crops, ResNet-50, four depths and both algebras, the median ratio of feature distances after and before the transformation is below one for heat and near one for hue in the **site** metric — the qualitative prediction of the generators — but the same ratio in the **consumer's** response sits slightly above one for both algebras at every depth (Table D.22). An input-space contraction is not a contraction of what a downstream consumer reads. That is why Theorem 4 is stated with $L$ in the consumer's seminorm and why a site's regime is reported as a distribution of typical stretch rather than read off the generator.""")

# ---------------------------------------------------------------- §6 depth
lit('sec_6_1_first',
    "What these curves measure, in the theory's terms, is how the piecewise-linear layer of §3.5 shows up layer by layer: the crossing share grows with depth (§3.5, Prop D), and so the achievable score falls.",
    "What these curves measure, in the theory's terms, is how the piecewise-affine structure of §3.3 shows up layer by layer: a transformation that visits more distinct local rules costs more to realise with one operator, and the unit-level decomposition of §6.6 measures where that cost lands.")

span('sec_6_3_mechanism',
     'The pairing is not an additional mechanism: it is the piecewise-linear layer of §3.5 read correctly.',
     'The ordering, not the algebra, is what the depth trend tracks.',
     r"""The pairing is not an additional mechanism: it is the piecewise-linear layer of §3.5 read correctly. Proposition D decomposes the effect of a transformation into a within-regime term and a term carried by the units whose gate changes, and a gate change is what makes two cells carry different local rules; whether one operator can serve them both is the joint system of Proposition C. The consumer-visible share of the change is a proxy for that sharing cost, not the cost itself. "A crossing counts only where the consumer reads it" is that statement, and the crossed design tests it. The mechanism numbers are consistent with it and quantitative: on ResNet-50 the consumer-visible crossing share rises with depth under both algebras — $0.220\to0.417\to0.627$ for hue and $0.252\to0.302\to0.516$ for heat, layers 2 to 4 — and across the six (site, algebra) points it is the consumer-visible share that orders the transfer score ($\rho=-0.886$), more tightly than the raw feature-level crossing share ($\rho=-0.83$) and far more tightly than the flip count, which does not order it at all. The ordering, not the algebra, is what the depth trend tracks.""")

lit('sec_6_3_consequence',
    'because the fraction of the transformation\'s effect that lands on crossed units grows — and a fixed linear map cannot express a function of the region index (§3.5).',
    'because the transformation visits more distinct local rules as depth grows, and whether one operator can serve them is the joint system of Proposition C (§3.5).')

span('sec_6_6_insert',
     'The next section turns that understanding into design.',
     'The next section turns that understanding into design.',
     r"""The next section turns that understanding into design.

### 6.6 Where the realisation error goes: a unit-level diagnostic

The two-source decomposition of §3.3 can be measured on real sites, and the measurement disciplines the mechanism. On a fit split of crops we fit (i) a per-channel affine law for each gate cell, (ii) a pooled per-channel affine law, and (iii) a linear operator in the top-$k$ principal subspace of the features; everything is then evaluated on a **held-out** split and normalised by the no-op displacement.

**Table 13.** Held-out realisation error, ResNet-50, COCO crops, hue $90°$ and heat $\sigma=2$. $R_{\mathrm{diag}}$ is the channel-diagonal operator (equivalently the pooled per-channel law), $S_{\mathrm{cell}}$ the per-cell law, *sharing* their difference, and $R_{\mathrm{full}}(k)$ the linear operator restricted to the top-$k$ principal subspace of the fit features.

| site | family | $R_{\mathrm{diag}}$ | $S_{\mathrm{cell}}$ | sharing | $R_{\mathrm{full}}(8)$ | $R_{\mathrm{full}}(16)$ | $R_{\mathrm{full}}(32)$ |
|---|---|---|---|---|---|---|---|
| layer1 | hue | $0.412$ | $0.412$ | $0.000$ | $0.394$ | $0.213$ | $\mathbf{0.101}$ |
| layer1 | heat | $0.084$ | $0.084$ | $0.000$ | $0.200$ | $0.099$ | $\mathbf{0.042}$ |
| layer2 | hue | $0.347$ | $0.347$ | $0.000$ | $0.921$ | $0.627$ | $0.402$ |
| layer3 | hue | $0.405$ | $0.405$ | $0.000$ | $1.691$ | $1.056$ | $0.846$ |
| layer4 | hue | $0.692$ | $0.650$ | $0.042$ | $1.186$ | $1.114$ | $1.053$ |
| layer4 | heat | $0.750$ | $0.696$ | $0.055$ | $1.365$ | $1.299$ | $1.243$ |

*Source: `scripts/263_two_term_heldout.py`; `results/two_term_heldout_resnet50.json`. 260 COCO crops, 60/40 fit/evaluation split, disjoint by construction.*

Two statements follow, and neither is the one an in-split measurement suggested. First, **the cost of sharing one operator across gate cells is negligible out of sample**: the per-cell and pooled channel laws agree to three decimals at the first three sites and differ only at the deepest. A decomposition evaluated inside the fit split reports a large sharing term because small cells are fitted where they are evaluated; held out, the term disappears, and the in-split variant (Table D.20) is retained only as the diagnostic it is. What grows with depth is the **pooled channel relation itself**. Second, **the benefit of cross-channel coupling reverses with depth**: a wide operator in the top principal subspace cuts the error substantially at the first layer, while at the last the full operator is worse than the diagonal restriction. Shallow layers let channels help each other; deep layers do not, and at depth the estimate is limited by the sample size available per channel.

That is the mechanism this section can support with one backbone, two algebras and one split: a degrading pooled channel relation and the loss of cross-channel coupling. It is a unit-level diagnostic of §3.3's decomposition, not the full-operator mechanism — these are per-cell affine and channel-diagonal families fitted at one site, whereas the depth decline of §6.1 is measured for a single global linear operator across layers. A dominant sharing cost is not what the held-out decomposition shows.""")

# ---------------------------------------------------------------- §7 interface
lit('sec_7_2_precision',
    'Training and anchoring protocols: method box, Appendix E.',
    r"""Training and anchoring protocols: method box, Appendix E.

The contribution, stated precisely: the rotation-block form comes from the known structure of the circle group, and the *measurement* decides which blocks are useful, how many are needed, and where the interface is valid; training maps existing features into those coordinates. "The action form is guided by measurement" is what the evidence supports; "the action form was discovered from the network" is not. Two constructions in this paper are also distinct and stay distinct in the wording: the local transporter of §5.3 has its generator and Fourier coefficients fitted and then frozen at inference (Appendix E, Box E1), while the carrier rotation used here is never fitted at all (Box E2).""")

span('thm5_main',
     '**Theorem 5 (carrier sufficiency)** makes the design licence explicit.',
     'it returns a negative answer for the semigroup.',
     r"""**Theorem 5 (carrier sufficiency).** Let the carrier be $c$, the synthesis $B$, the target $z'=BAc+w'$ with $\|w'\|_{L^2}\le\delta'$, the read-in $E_\theta$ (any measurable map, in particular the learned MLP), and let the interface apply $A_{\mathrm{int}}$ and decode by $B$. Write $e(z)=E_\theta(z)-c$ for the read-in error. Then the identity
$$BA_{\mathrm{int}}E_\theta(z)-z'=BA_{\mathrm{int}}\,e(z)+B(A_{\mathrm{int}}-A)c-w'$$
holds exactly, and hence, with no cross-term cancellation assumed,
$$\big\|BA_{\mathrm{int}}E_\theta(z)-z'\big\|_{L^2}\;\le\;\|BA_{\mathrm{int}}\|\,\|e\|_{L^2}+\|B(A_{\mathrm{int}}-A)\|\,\|c\|_{L^2}+\delta'.$$
*Proof.* Add and subtract $BA_{\mathrm{int}}c$ and $BAc$; the first term is the read-in error, the second the action mismatch, the third the target's tail. The triangle inequality gives the bound. $\square$

Three consequences fix how the bound is to be read. *It applies to the interface as built*: $E_\theta$ need not be linear, and the linear read-in is the special case in which $\|e\|$ is written as an operator norm. *The three terms are read-in error, action mismatch, and target tail* — only the first is removed by training, and the second vanishes when the interface uses the true carrier action. *Order of the mismatch term:* the bound is first order in $\|A_{\mathrm{int}}-A\|$, and the squared error is therefore second order in the plane misalignment; the measured quantity is the error norm, for which the log-log slope in the tilt angle is $0.99$. If the carrier tail is normalised as an energy fraction, $\mathbb E\|w\|^2\le\varepsilon_{\mathrm{tail}}^2\mathbb E\|z\|^2$, then the measured $12$–$16\%$ energy fraction corresponds to $\varepsilon_{\mathrm{tail}}\approx0.35$–$0.40$, not to $0.12$–$0.16$.

The closure criterion of §4.6 enters separately, and its scope is worth stating precisely: Proposition G bounds what a **fixed linear map on the measured features** can achieve, and it is that class — not the interface — which the measured closure defect limits. The interface escapes the bound exactly because $E_\theta$ is learned and may be nonlinear; what it cannot escape is the *target's* tail and any mismatch between the action it applies and the action the carrier actually has. This is the sense in which the construction is sufficient: the action is fixed by measurement, the read-in is learned, and the two structural conditions — closure of the retained blocks and alignment of the interface planes — are checkable in advance on the measured organisation. The theorem is conditional, its conditions are measurable, and the ordering it predicts — carrier error tracking the tail and the plane misalignment site by site — is not tested here; the bound is sufficient rather than tight. **The framework's *theory* is algebra-general; its *interface* is not.** The hue interface exists because both conditions hold there (§4.3); the dissipative family fails the closure premise in every basis tested (§4.7), which is why it is carried in channel space rather than by the canonical spatial action and why its chains degrade monotonically. A carrier-sufficiency theorem with measurable premises is a filter rather than a promise: it says which algebras to attempt to compile, and on this instance it returns a negative answer for the semigroup.""")

lit('table10_composition_row',
    r'| composition to a never-fitted parameter | $T_F$ at the composite | direct fit at that parameter | composed $\ge$ direct (Table 5) | plain CNN stage 1, three seeds |',
    r'| composition to a never-fitted parameter | $T_F$ at the composite | direct fit at that parameter | composed $0.844$ vs direct $0.807$ at $\sigma=1.803$ | plain CNN stage 1, seed 0; never-fitted composite (Table D.11) — the fixed-total chains of Table 5, three seeds, score $0.451$–$0.488$ against a direct $0.511$ |')

lit('table10_source',
    '*Sources: `results/hue_probe_app.json`, `results/multipath_composition.json`, `results/detector_o8_multiseed.json`, `results/region_rho_demo.json`, `results/code3d_sv_lowext.json`.; produced by scripts 67, 231, 156, 109, 112.*',
    '*Sources: `results/hue_probe_app.json`, `results/multipath_composition.json`, `results/heat_tables.json` (never-fitted composite), `results/detector_o8_multiseed.json`, `results/region_rho_demo.json`, `results/code3d_sv_lowext.json`.; produced by scripts 67, 231, 176, 156, 109, 112.*')

lit('table3_source',
    '*Source: `results/hue_prediction_demo.json`, `action_fragmentation.json`, `causal_cifar_summary.json`; `locality_*`, `blockrot`, `intfreq_*` for the rejected families; produced by scripts 64, 65, 136.*',
    '*Source: `results/hue_prediction_demo.json`, `action_fragmentation.json`, `causal_cifar_summary.json`; the coordinate-monotone-versus-linear comparison is `results/family_capacity.json` (script 257); `locality_*`, `blockrot`, `intfreq_*` for the rejected families; produced by scripts 64, 65, 136.*')

# ---------------------------------------------------------------- §8 discussion, §9 conclusion
lit('sec_8_2_thm3',
    'the capacity boundary of per-dimension families (Thm 3) constrains any fixed-basis family on any non-planar orbit',
    'the statement about coordinate-monotone families (Thm 3) is a restricted-class result about closed orbits')

lit('sec_8_2_closure',
    'and carrier sufficiency (Thm 5) is stated for any site meeting two measurable premises, which is why it can return a *negative* answer for an algebra — as it does for the dissipative family.',
    'the closure criterion (Props F–G) is stated for any carrier and action, which is why it can be applied to a measured organisation as a design condition; and carrier sufficiency (Thm 5) is stated for any site meeting its premises, which is why it can return a *negative* answer for an algebra — as it does for the dissipative family.')

lit('sec_8_5_heading',
    '**8.5 Threats to validity, stated rather than implied.** Five matter.',
    '**8.5 Limits and threats to validity.** Six matter.')

lit('sec_8_5_item_v',
    '(v) *Partial algebra coverage.* Colour is carried through the full protocol;',
    '(v) *Closure is a design criterion, not a diagnosis.* On measured features the synthesis has full column rank, the exact carrier model is closed automatically, and the residual failure of a global fit is attributable to cross-content inconsistency and tail rather than to non-closure; the criterion\'s role is to say which coordinate sets a compiled interface may keep. (vi) *Partial algebra coverage.* Colour is carried through the full protocol;')

lit('sec_9_theory',
    "*Theory:* an induced action exists exactly when the encoder's fibres survive the transformation, existence already delivers the composition law, a linear realisation exists exactly when the kernel is preserved with defect governed by the mixing ratio, and on the rectifier layer the failure to realise decomposes into a within-regime term and a crossing term whose concentration — not the number of flips — bounds what any fixed linear operator can achieve.",
    "*Theory:* an induced action exists exactly when the encoder's fibres survive the transformation, existence already delivers the composition law, a linear realisation exists exactly when the kernel is preserved with defect governed by the mixing ratio in the metric the representation induces, on the rectifier layer the failure to realise decomposes into a within-regime term and a term carried by the cells whose rule changes — the change being the source of rule diversity, not the obstruction — and a carrier admits a fixed action exactly when its retained blocks are closed, in which case the optimal map is the carrier action seen through the truncation.")

lit('sec_9_empirical',
    'and the price of a chain is set by whether the truth action is contractive, isometric, or expansive.',
    "and the price of a chain is governed by the truth action's Lipschitz constant in the consumer's seminorm, whose regime has to be measured rather than read off the generator.")

# ---------------------------------------------------------------- Appendix A
lit('appA_thm2_row',
    r'| Thm. 2 | $1-T_{RMS}=\kappa/\sqrt{1+\kappa^2}$ with $\kappa=\lVert C\rVert_F/\lVert A-I\rVert_F$ | **Verified** |',
    r'| Thm. 2 | $1-T_{RMS}=\kappa_W/\sqrt{1+\kappa_W^2}$ with $\kappa_W=\lVert WC\rVert_F/\lVert W(A-I_{\mathcal R})\rVert_F$; $\kappa$ when $W$ is an isometry on the retained subspace | **Verified** |')

lit('appA_propC_row',
    r'| Prop. C | multi-region compatibility is a joint affine system | **Theorem** | Appendix B.4 | piecewise-linear, no crossing |',
    r'| Prop. C | region-transition criterion: a fixed linear realisation exists iff the joint system over the realised transition cells is consistent (Cor. C.1 for the no-crossing case) | **Theorem** | Appendix B.4 | piecewise-linear sites |')

lit('appA_propD_row',
    r"| Prop. D | $z'-z=G\odot\Delta a+\Delta_{\mathrm{flip}}$, a jump at flipped gates | **Theorem** | Appendix B.4 | rectifier sites |",
    r"| Prop. D | $z'-z=G\odot(a'-a)+(G'-G)\odot a'$; gate changes are the source of rule diversity, not the obstruction | **Theorem** | Appendix B.4; two counterexamples in both directions | rectifier sites |")

lit('appA_propE_row',
    r"| Prop. E | $T_F\le1-\lVert\mathrm{Proj}_F\Delta_{\mathrm{flip}}\rVert_F/\lVert\mathrm{Proj}_F(z'-z)\rVert_F$ | **Empirical bound** (exact form proved for linear consumers) | 6/6 site–algebra points; slack = within-regime non-linearity | fixed-linear class |",
    r"""| Prop. E | $T_F\le1-\lVert\mathrm{Proj}_F\Delta_{\mathrm{flip}}\rVert_F/\lVert\mathrm{Proj}_F(z'-z)\rVert_F$ | **Empirical bound** (exact form proved for linear consumers) | 6/6 site–algebra points; slack = within-regime non-linearity | fixed-linear class |
| Prop. F, G | closure criterion $A\ker B\subseteq\ker B$; optimal linear realisation $K^\star=BAB^\dagger$ with residual $\|BAP_{\ker B}\|_F^2$ (coloured carrier: after whitening) | **Theorem** + verified | Appendix B.3b; 40 random selections, gap $9\times10^{-16}$ | carrier and action; a **design** condition |""")

lit('appA_thm3_row',
    r'| Thm. 3 | capacity boundary of per-dimension families; error $\ge\sigma_3^2(1-\varepsilon)$ | (a),(b) **Theorem**; (c) stated with slack | Appendix B.5.1; measured third component $\approx10\%$ of orbit energy | fixed-basis per-dimension monotone families |',
    r'| Thm. 3 | a closed orbit admits no exact coordinate-monotone realisation; per-coordinate cost bound | **Theorem** | Appendix B.5.1; best fit measured at $L^2=998$ against $924$ for the best linear map | monotone parameter families in a fixed basis |')

lit('appA_thm5_row',
    r'| Thm. 5 | carrier sufficiency: error $\le\varepsilon_{\mathrm{tail}}+$ (term linear in plane misalignment) | **Theorem**; premises measured | Appendix B.5.3; $\varepsilon_{\mathrm{tail}}\approx12$–$16\%$ at $K=2$, plane $\cos\ge0.96$ | sites with band-limited orbits and shared planes |',
    r'| Thm. 5 | carrier sufficiency: three-term $L^2$ bound on read-in error, action mismatch and target tail | **Theorem**; premises measured | Appendix B.5.3; tail energy $12$–$16\%$ ($\varepsilon_{\mathrm{tail}}\approx0.35$–$0.40$), mismatch exponent $0.99$ | sites satisfying closure and tail bounds |')

lit('appA_measured_add',
    '*Source: statements and numbers as in the main text, each also traced in Tables B.1 and B.4.',
    r"""| The sharing term of the two-source decomposition is an in-split artefact | Measured (negative, held-out) | per-cell and pooled channel laws agree to $0.000$ at the first three sites and differ by $0.042$/$0.055$ at the last; what degrades is the pooled relation ($0.412\to0.347\to0.405\to0.692$ for hue) | ResNet-50, held-out split |
| Input-space metric distortion does not transfer to the consumer | Measured (negative) | site metric $0.80$–$0.99$ (heat) and $0.97$–$1.07$ (hue); consumer response $1.02$–$1.06$ for both | ResNet-50, four depths |

*Source: statements and numbers as in the main text, each also traced in Tables B.1 and B.4.""")

# ---------------------------------------------------------------- Appendix B
span('appB_thm2',
     '**Theorem 2 (defect law).** *Under Assumptions B and C let $\\rho^\\star$ minimise',
     r'on the sweep of Table 2 the two differ by at most $0.0011$. $\square$',
     r"""**Theorem 2 (defect law, in the induced metric).** *Under Assumptions B and C let $\rho^\star$ minimise $\mathbb E\lVert\rho z(x)-z(gx)\rVert^2$ over the population, and let $T_{RMS}$ be the RMS-criterion transfer of §3.3. Then*
$$1-T_{RMS}=\frac{\lVert WC\rVert_F}{\sqrt{\lVert W(A-I_{\mathcal R})\rVert_F^2+\lVert WC\rVert_F^2}}=\frac{\kappa_W}{\sqrt{1+\kappa_W^2}},\qquad \kappa_W:=\frac{\lVert WC\rVert_F}{\lVert W(A-I_{\mathcal R})\rVert_F},$$
*and when $W$ is an isometry on the retained subspace this reduces to $\kappa=\lVert C\rVert_F/\lVert A-I\rVert_F$ with the optimum at $\rho^\star=A$; otherwise the optimum is $W$-conjugate.* *Proof.* By Lemma B.1 the objective is $\mathbb E\lVert(\rho W-WA)x_R-WCx_K\rVert^2$. On $\mathcal R$ the operator $\rho W-WA$ acts and $x_R$ ranges over a dense subset, so the minimiser restricts to $\rho W|_{\mathcal R}=WA$; under isotropy $x_R\perp x_K$ and the normal equations decouple. The residual is $WCx_K$ and the displacement is $W(A-I)x_R+WCx_K$, so $\mathbb E\lVert WCx_K\rVert^2=\lVert WC\rVert_F^2$ and $\mathbb E\lVert W(A-I)x_R\rVert^2=\lVert W(A-I)\rVert_F^2$; dividing gives the displayed form. The mean-ratio transfer $T_F=1-\mathbb E\lVert\rho z-z'\rVert/\mathbb E\lVert z-z'\rVert$ is a different statistic and satisfies the same identity only up to the fluctuation of the norm ratio; on the sweep of Table 2 the two differ by at most $0.0011$. $\square$

*Why the metric appears.* Writing the law without $W$ assumes the encoder is an isometry on the retained subspace — true after whitening, false in general. A coordinate the encoder scales up counts more in $\lVert WC\rVert_F$, so a defect law calibrated at one site need not transfer to another. The qualitative content is metric-free: $C$ is the only block that obstructs existence (Theorem 1) and the only one that enters the defect.""")

span('appB_B3b',
     '#### B.4 The piecewise-linear layer (Propositions C, D, E)',
     '#### B.4 The piecewise-linear layer (Propositions C, D, E)',
     r"""#### B.3b Carrier closure and optimal linear realisation

**Proposition F (closure criterion).** *For a carrier map $B$ and a carrier action $A$, there exists a fixed linear $K$ with $KB=BA$ if and only if $A\ker B\subseteq\ker B$.* *Proof.* $KB=BA$ is solvable iff $BA$ vanishes on $\ker B$; a linear map is determined on a subspace only if it maps that subspace into $\mathrm{ran}\,B$. $\square$

**Proposition G (optimal linear realisation).** *Let $\mathbb E[cc^\top]=I$, $z=Bc$ and $z'=BAc$. Then $K^\star=BAB^\dagger$ is a least-squares minimiser and*
$$\min_K\mathbb E\|Kz-z'\|^2=\|BA P_{\ker B}\|_F^2,$$
*which is zero exactly under Proposition F. For a coloured carrier, $\mathbb E[cc^\top]=\Sigma_c$, the residual is $\|BA\Sigma_c^{1/2}P_{\ker(B\Sigma_c^{1/2})}\|_F^2$ and the minimiser is $K^\star=BA\Sigma_c B^\top(B\Sigma_cB^\top)^\dagger$, with $\Sigma_c^{1/2}$ the symmetric square root: correlation lets a retained coordinate predict a discarded one, so the projector is taken after whitening, and the two forms coincide only for $\Sigma_c\propto I$.* *Proof.* $K^\star z-z'=BA(B^\dagger B-I)c=-BAP_{\ker B}c$ because $B^\dagger B$ is the orthogonal projector onto the row space of $B$; taking the expectation with $\mathbb E[cc^\top]=I$ gives the squared Frobenius norm, and the coloured case follows by whitening $c\mapsto\Sigma_c^{-1/2}c$. $\square$

*Corollary (coordinate selection).* If $B=P$ selects coordinates and the carrier is orthogonal along the orbit, $K^\star=PAP^\top$: the optimal fixed linear map is the carrier action projected onto the retained coordinates. Verified over forty random selections of a six-dimensional carrier, where the gap to $PAP^\top$ was $9\times10^{-16}$ and the closure verdict coincided with the sign of the residual in every case (`results/carrier_optimality_identity.json`, script 259).

*Corollary (conjugate pairs).* For a full continuous rotation action written in one two-dimensional block per frequency, with selection along those coordinates, closure holds iff each block is retained or dropped as a whole. The restriction is part of the statement: for a single fixed angle, or when frequencies repeat and their blocks may mix, the relevant object is the invariant subspace rather than the pair. The instantiation on measured features, including the dropped-coordinate variant, is Table D.21.

#### B.4 The piecewise-linear layer (Propositions C, D, E)""")

span('appB_B4',
     '**Proposition C (multi-region compatibility).** *Suppose',
     r'$\rho(\text{flip depth},T_F)=\rho(\text{crossing share},T_F)=-0.83$.',
     r"""**Proposition C (region-transition criterion).** *Let the site be affine on each visited region, $\phi=A_rx+b_r$, let $T$ be the input transformation, and let $(r,s)$ range over the realised transitions, each covering a set with non-empty interior or spanning its region. Then a fixed linear $\rho$ with $\rho\phi=\phi\circ T$ on the visited cells exists if and only if the joint system $\rho A_r=A_sT,\ \rho b_r=b_s$ is consistent over all realised $(r,s)$.* *Proof.* On a cell both $\rho\phi(x)$ and $\phi(Tx)$ are affine in $x$; an identity between affine maps on a set with non-empty interior extends to the maps, so the system is necessary, and a solution clearly suffices. $\square$

**Corollary C.1 (no crossing).** *If $T$ leaves the gate pattern of each visited region unchanged, the realised transitions are the diagonal cells and the system reduces to $\rho A_r=A_rT,\ \rho b_r=b_r$.* The special case is the one in which each source region maps into itself; consistency with crossings is independent of it, as the two counterexamples of §3.5 show.

For a single visited region the system is $\rho A=A\tau,\ \rho b=b$, which is Theorem 1's kernel condition rewritten — the linear layer is the single-region case of this one.

*Consequence.* **Even with no crossing at all, the multi-region structure alone can obstruct a global linear action**: the visited regions must be simultaneously conjugate to the transformation. This is why enlarging the operator family buys nothing — measured, a ridge-plus-$3\times3$-residual family and the global linear family agree to within $0.087$ at every site (§5.2).

**Proposition D (the decomposition, and where rule diversity comes from).** *With $z=\mathrm{ReLU}(a)$, $a'=a+\Delta a$, $G=\mathbb 1[a>0]$ and $G'=\mathbb 1[a'>0]$,*
$$z'-z=G\odot(a'-a)+(G'-G)\odot a',$$
*and on a unit whose gate changes the second term is $+a'$ (off$\to$on) or $-a'$ (on$\to$off).* *Proof.* $z'=G'\odot a'=G\odot a'+(G'-G)\odot a'$ and $z=G\odot a$. $\square$

Two consequences, and the second governs how the measurements are read. First, the within-regime term is linear in $\Delta a$; the gate-change term has magnitude $|a'|$ on the changed units, so it is not a constant that survives a shrinking perturbation. Second, **a gate change is not itself an obstruction**: it is the origin of a change in the local affine rule, and whether a single fixed linear operator can satisfy all the rules the transformation visits is decided by the joint system of §3.3, which may be consistent with every gate changed, or inconsistent with no gate changed in the source region. Both directions have explicit two-dimensional counterexamples in §3.5. Measured (script 247; ResNet-50, three depths $\times$ two algebras): $\rho(\text{flip count},T_F)=+0.49$ (n.s.) against $\rho(\text{flip depth},T_F)=\rho(\text{crossing share},T_F)=-0.83$.""")

span('appB_thm3',
     '**Theorem 3 (capacity boundary of per-dimension families).** *(a) [Proved]',
     r'rather than as a sharp constant.',
     r"""**Theorem 3 (a closed orbit has no exact coordinate-monotone realisation).** *A family $\{G_t\}$ whose members are coordinate-wise monotone in a fixed basis, with $(t,u)\mapsto f_{i,t}(u)$ non-decreasing in both arguments, realises an orbit only if every coordinate of that orbit is monotone in the parameter (Lemma B.7); consequently a closed non-degenerate orbit admits no exact realisation by such a family. Quantitatively, every coordinate that decreases by $A$ over intervals of length at least $\ell$ contributes $L^2$ error at least $A^2\ell/4$ (Lemma B.8).* The empirical content is the comparison in Table 3: the class is rejected both by the theorem and by its measured best fit on the reference orbit. *Scope.* The statement is about monotone parameter families, which is what the lemmas prove; an aggregate bound in terms of the third principal value of the orbit covariance does not follow from them and is not claimed here.""")

span('appB_thm5',
     '**Setting.** At the site, decompose the feature into harmonic components',
     'the dissipative family violates in every basis tested.',
     r"""**Setting.** At the site, let the carrier be $c$, the synthesis $B$, the target $z'=BAc+w'$ with $\lVert w'\rVert_{L^2}\le\delta'$, the read-in $E_\theta$ (any measurable map, in particular the learned MLP), and let the interface apply $A_{\mathrm{int}}$ and decode by $B$. Band-limitation and plane sharing are the two premises the construction rests on and both are measurable, but the theorem below does not need them as hypotheses: it is an identity plus a triangle inequality for whatever $B$, $A$ and $E_\theta$ are.

**Theorem 5 (carrier sufficiency, three-term bound).** *Write $e(z)=E_\theta(z)-c$ for the read-in error. Then*
$$BA_{\mathrm{int}}E_\theta(z)-z'=BA_{\mathrm{int}}\,e(z)+B(A_{\mathrm{int}}-A)c-w'$$
*holds exactly, and hence*
$$\big\|BA_{\mathrm{int}}E_\theta(z)-z'\big\|_{L^2}\le\|BA_{\mathrm{int}}\|\,\|e\|_{L^2}+\|B(A_{\mathrm{int}}-A)\|\,\|c\|_{L^2}+\delta'.$$
*Proof.* Add and subtract $BA_{\mathrm{int}}c$ and $BAc$; the first term is the read-in error, the second the action mismatch, the third the target's tail. The triangle inequality gives the bound, with no cross-term cancellation assumed. $\square$

*Three consequences.* The bound applies to the interface as built: $E_\theta$ need not be linear, and a linear read-in is the special case in which $\lVert e\rVert$ is written as an operator norm. Its three terms are read-in error, action mismatch and target tail; only the first is removed by training, and the second vanishes when the interface uses the true carrier action. The mismatch term is first order in $\lVert A_{\mathrm{int}}-A\rVert$, so the squared error is second order in the plane misalignment, while the measured quantity is the error norm, whose log-log slope in the tilt angle is $0.99$ (`results/carrier_sufficiency.json`, script 257). If the carrier tail is normalised as an energy fraction, $\mathbb E\lVert w\rVert^2\le\varepsilon_{\mathrm{tail}}^2\mathbb E\lVert z\rVert^2$, the measured $12$–$16\%$ energy fraction corresponds to $\varepsilon_{\mathrm{tail}}\approx0.35$–$0.40$, not to $0.12$–$0.16$. The closure obstruction of B.3b enters separately and constrains a *fixed linear map on the measured features*, a class the learned, possibly nonlinear read-in escapes; what it cannot escape is the target's tail and any mismatch between the action it applies and the action the carrier has.""")

lit('appB_status_propC',
    r'| Proposition C (multi-region compatibility) | Proved | piecewise-linear sites, no crossing | B.4 |',
    r'| Proposition C (region-transition criterion) | Proved | piecewise-linear sites, realised transitions | B.4 |')

lit('appB_status_propD',
    r'| Proposition D (crossing decomposition) | Proved | rectifier sites | B.4 |',
    r'| Proposition D (decomposition and rule diversity) | Proved | rectifier sites | B.4 |')

lit('appB_status_thm3',
    r'| Theorem 3 (capacity boundary) | (a),(b) Proved; (c) stated with slack | fixed-basis per-dimension monotone families | B.5.1 |',
    r'| Theorem 3 (no exact coordinate-monotone realisation of a closed orbit) | Proved | monotone parameter families in a fixed basis | B.5.1 |')

lit('appB_status_thm5',
    r'| Theorem 5 (carrier sufficiency) | Proved; premises measured | sites satisfying band-limitation + plane sharing | B.5.3 |',
    r"""| Proposition F (closure criterion), Proposition G (optimal linear realisation) | Proved + verified | carrier and action; any site | B.3b |
| Theorem 5 (carrier sufficiency, three-term bound) | Proved; premises measured | sites satisfying closure and tail bounds | B.5.3 |""")

# ---------------------------------------------------------------- Appendix D
lit('appD_intro',
    'This appendix carries the per-backbone, per-site and per-cell grids behind Tables 1, 3, 4, 5, 6, 7, 8, 9, 10, 11 and 12 of Sections 4–7.',
    'This appendix carries the per-backbone, per-site and per-cell grids behind Tables 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and 13 of Sections 4–7.')

span('appD_D8',
     '*Source: `results/crossing_consumer_resnet50.json`. Reproduces §3.5 (Props. D–E); the $-0.83$ quoted there is the raw feature crossing share.*',
     '*Source: `results/crossing_consumer_resnet50.json`. Reproduces §3.5 (Props. D–E); the $-0.83$ quoted there is the raw feature crossing share.*',
     r"""*Source: `results/crossing_consumer_resnet50.json`. Reproduces §3.5 (Props. D–E); the $-0.83$ quoted there is the raw feature crossing share.*

#### D.8 Realisation-error and metric diagnostics

**Table D.20.** The two sources of realisation error measured *inside the fit split* — the in-split diagnostic whose held-out version is Table 13 of the main text: the within-cell term, the cost of sharing one operator across the realised transition cells, and the share of the within-cell term contributed by the off$\to$on cell (where the source activation is zero while the target varies).

| site | hue within-cell | hue sharing (share of total) | heat within-cell | heat sharing (share of total) | off$\to$on share of within (hue / heat) | units |
|---|---|---|---|---|---|---|
| `layer1` | 0.0836 | 0.2000 (71\%) | 0.1583 | 0.4655 (75\%) | 6\% / 17\% | 1135.0 |
| `layer2` | 0.0789 | 0.2085 (73\%) | 0.1104 | 0.3485 (76\%) | 8\% / 15\% | 2615.0 |
| `layer3` | 0.0535 | 0.1697 (76\%) | 0.0669 | 0.2458 (79\%) | 13\% / 20\% | 5640.0 |
| `layer4` | 0.0131 | 0.3133 (96\%) | 0.0112 | 0.3622 (97\%) | 59\% / 63\% | 8834.0 |

*Source: `results/two_term_decomposition_resnet50.json`, produced by `scripts/260_two_term_decomposition.py` (ResNet-50, 100 COCO crops, six sampled locations per site, hue $90°$ and heat $\sigma=2$); values are mean squared errors per unit. Evaluated inside the fit split, the sharing term looks dominant; evaluated on held-out data (Table 13) it is $\approx0$, which is why Table 13 is the diagnostic the paper reports.*

**Table D.21.** The closure criterion instantiated on measured features: the carrier of the known input hue against the estimated synthesis $B$, with the relative reconstruction tail, the closure defect of $B$ against $A_\Delta$, and the held-out relative error of a global linear map fitted on half the hues; the last three columns repeat the analysis with the $\sin2h$ coordinate dropped.

| site | tail | closure defect | held-out global fit | tail (no $\sin2h$) | defect (no $\sin2h$) | held-out (no $\sin2h$) |
|---|---|---|---|---|---|---|
| `yolo11n_y3` | 0.046 | 3.2e-15 | 0.282 | 0.060 | 2.0e-15 | 0.334 |
| `resnet18_l2` | 0.064 | 3.5e-15 | 0.413 | 0.087 | 2.3e-15 | 0.395 |

*Source: `results/real_site_closure.json`, produced by `scripts/261_real_site_closure.py` (dense5 orbits at the YOLO11n $y3$ and ResNet-18 $l2$ sites, $37°$ shift, half the hues fitted and the other half held out). The closure defect vanishes because the synthesis has full column rank, which is why the residual failure of the global fit there is attributed to cross-content inconsistency and tail rather than to non-closure (§4.6).*

**Table D.22.** Metric distortion of the truth action in the site metric and in the consumer's response, ResNet-50, four depths, both algebras: median and p90 of the pairwise ratio of feature distances after and before the transformation, over 120 COCO crops with a 60-crop subsample; the consumer column is the same at every site because the consumer is the network's own output.

| site | hue: site metric (median / p90) | heat: site metric (median / p90) | hue: consumer response | heat: consumer response |
|---|---|---|---|---|
| layer1 | $1.067$ / $1.376$ | $0.847$ / $0.989$ | $1.056$ | $1.015$ |
| layer2 | $0.995$ / $1.094$ | $0.800$ / $0.900$ | $1.056$ | $1.015$ |
| layer3 | $0.974$ / $1.035$ | $0.877$ / $0.946$ | $1.056$ | $1.015$ |
| layer4 | $1.027$ / $1.180$ | $0.990$ / $1.140$ | $1.056$ | $1.015$ |

*Source: `scripts/258_measure_lipschitz.py`; `results/lipschitz_resnet50.json`. These are distributions of typical stretch, not a uniform bound; the supremum over the relevant state domain can differ, as can error cancellation along a chain and the changing set of visited states, which is why Theorem 4's $L$ is measured in the consumer's seminorm rather than read off the generator (§5.4).*""")

# ---------------------------------------------------------------- Appendices E, G
span('appE_boxE2_scope',
     'which are measured. One caveat belongs to the runtime figure of Box E4:',
     'which are measured. One caveat belongs to the runtime figure of Box E4:',
     "which are measured. The two constructions differ in kind: the local transporter of Box E1 fits a generator and its Fourier coefficients and freezes them at inference, whereas the action here is never fitted and only the read-in is learned. One caveat belongs to the runtime figure of Box E4:")

lit('sec_2_4_thm3_gloss',
    'we do not assume a linear realisation exists, we measure when it does (§5.2), prove a capacity boundary when it does not (Thm 3), and build the working object from the measurements (§7).',
    'we do not assume a linear realisation exists, we measure when it does (§5.2), prove that the natural per-dimension class cannot realise a closed orbit (Thm 3), and build the working object from the measurements (§7).')

lit('appG_gruver_clause',
    'and our Proposition D shows the local quantity it measures is the crossing term |',
    'and our Proposition D shows the local quantity it measures belongs to the gate-change term, whose role in the failure is decided by the region-transition criterion |')

lit('sec_3_5_gate_change_clause',
    "The gate-change term has magnitude $|a'|$ on the changed units, so it is not a constant that vanishes with the perturbation, and no fixed linear operator represents a function of the discrete region index.",
    "The gate-change term has magnitude $|a'|$ on the changed units, so it is not a constant that vanishes with the perturbation, and it carries a dependence on the discrete region index whose reachability by one fixed linear operator is the joint system of Proposition C rather than a matter of counting flips.")

lit('appB_B4_section_ref',
    'is decided by the joint system of §3.3, which may be consistent with every gate changed',
    'is decided by the joint system of §3.5, which may be consistent with every gate changed')

span('appG_koopman_row',
     '| **Dynamical / Koopman-style linearisation** |',
     'is what makes the truncated form work |',
     r"""| **Dynamical / Koopman-style linearisation** | Koopman-style embedding work as surveyed in §2.4 | when does a nonlinear system admit an invariant finite-dimensional subspace on which the evolution is linear? | the idea that a linear operator on an embedding can stand for a nonlinear action, which our band-limited global family instantiates in truncated form | the subspace is *designed or learned* and the dynamics are temporal; the line does not ask whether a *given, frozen* representation admits a linear realisation, nor what obstructs one | the realisation question is *decided* for the given representation, not assumed: existence is an iff condition (Thm 1), the closure criterion decides which carrier coordinates can carry a fixed action (Props F–G), the capacity of the natural per-dimension class is bounded before fitting (Thm 3), and the failure is attributed to a measurable term (Prop D) | a truncated Koopman operator on features is one of our rejected global families (§5.2) unless the orbit is band-limited — the measured spectrum ($k\le2$ at $84$–$88\%$) is what makes the truncated form work |""")


# ================================================================================
# Consistency fixes from the adversarial audit of the assembled v17 (round 2).
# These run after the integration ops above, so their markers match v17's text.
# ================================================================================

span('fix_propE_main',
     '**Proposition E (an achievable-score bound).**',
     'The measured version is therefore an **empirical bound**, not a theorem.',
     r"""**Proposition E (achievable score of the fixed-linear class).** Let $F$ be the consumer, $\lVert\cdot\rVert_F$ the seminorm its response induces, and $d=z'-z$. The best score attainable by any family whose members are fixed linear maps of $z$ is
$$T_F^{\max}=1-\frac{\lVert\mathrm{Proj}_F d^\perp\rVert_F}{\lVert\mathrm{Proj}_F d\rVert_F},$$
where $d^\perp$ is the residual of the $L^2$ projection of $d$ on the consumer-visible linear span of $z$: the class reproduces the part of the change that is linearly predictable from $z$ and nothing else. *Status.* This exact form is proved for linear consumers and verified at six (site, algebra) points; no claim is made here for families outside the fixed-linear class.

**The gate-change share is a proxy, not a bound.** Writing the gate-change term of Proposition D as $\Delta_{\mathrm{flip}}:=(G'-G)\odot a'$, its consumer-visible share is a good proxy for the slack above whenever that term is the part of $d$ not linearly predictable from $z$ — the generic case, and the reason the share orders the measured transfer scores of §6.3. The proxy is an identity in neither direction: in the counterexample above every gate changes and the slack is zero, and a consumer blind to the change has zero visible share whatever the slack. The measured association is what the proxy licenses; the two-source decomposition of §3.3 is what the failure actually consists of.""")

span('fix_propE_appB',
     '**Proposition E (achievable-score bound).**',
     "the six-point check is the paper's evidence that neither dominates at the sites measured.",
     r"""**Proposition E (achievable score of the fixed-linear class).** *Let $F$ be the consumer, $\lVert\cdot\rVert_F$ the seminorm its response induces, and $d=z'-z$. For a linear consumer the identity*
$$T_F^{\max}=1-\frac{\lVert\mathrm{Proj}_F d^\perp\rVert_F}{\lVert\mathrm{Proj}_F d\rVert_F}$$
*is exact, with $d^\perp$ the residual of the $L^2$ projection of $d$ on the consumer-visible linear span of $z$: the class of fixed linear maps can reproduce any component of $d$ that is linearly predictable from $z$, and nothing else.* *Proof.* Write $d=d^{\parallel}+d^\perp$ with $d^{\parallel}$ in the visible span of $z$; the map $K=d^{\parallel}$ is attainable and leaves the residual $d^\perp$, which no member of the class can reach, so the optimum is $d^{\parallel}$ and the displayed ratio is the score attained. $\square$

*Status.* Proved for linear consumers, verified at six (site, algebra) points. No claim is made for families outside the fixed-linear class.

*The gate-change share is a proxy.* With $\Delta_{\mathrm{flip}}:=(G'-G)\odot a'$ the gate-change term of Proposition D, its consumer-visible share approximates the slack when that term is the part of $d$ not linearly predictable from $z$. It is an identity in neither direction: the all-gates-change counterexample of §3.5 has zero slack, and a consumer blind to the change has zero visible share whatever the slack. The six-point check is the paper's evidence that the proxy is informative at the measured sites; the two-source decomposition of §3.3 is the statement about the failure itself.""")

lit('fix_thm5_4_3',
    'band-limitation and plane sharing are exactly the two conditions of Theorem 5, measured here before §7 builds on them (Table 12).',
    'band-limitation and plane sharing are the two measured conditions the interface\'s construction uses, quantified here before §7 builds on them (Table 12).')

span('fix_thm5_7_2',
     "the licence for the fixed action is Theorem 5's premises, which are measured. Training and anchoring protocols",
     "the licence for the fixed action is Theorem 5's premises, which are measured.",
     "the licence for the fixed action is that the two conditions the construction uses — the retained carrier is closed and its planes align with the measured ones — hold at the site, and both are measured (§4.3, §7.3).")

lit('fix_thm5_heading',
    '### 7.3 Why the construction is sufficient when the measured conditions hold',
    '### 7.3 What bounds the interface: the three error terms')

lit('fix_thm5_identity',
    'Three consequences fix how the bound is to be read.',
    'The theorem is an identity, not a conditional bound: it holds for any carrier, action and read-in, and what the measurements supply is the *size* of its three terms rather than a hypothesis of the theorem. Three consequences fix how the bound is to be read.')

lit('fix_thm5_bridge',
    'A carrier-sufficiency theorem with measurable premises is a filter rather than a promise: it says which algebras to attempt to compile, and on this instance it returns a negative answer for the semigroup.',
    'A carrier-sufficiency theorem with measurable premises is a filter rather than a promise: it says which algebras to attempt to compile, and on this instance it returns a negative answer for the semigroup. One boundary of the statement belongs here: the theorem bounds a *feature-space* error, while the interface\'s read-out figures are measured in degrees, and the link between the two is the decoder\'s local sensitivity rather than a second theorem. What the decomposition supplies for the capability claims is the attribution: of the three terms, only the read-in error is removed by training, and the few-degree read-out errors of §7.4 are what remains.')

lit('fix_appA_thm5_scope',
    r'| Thm. 5 | carrier sufficiency: three-term $L^2$ bound on read-in error, action mismatch and target tail | **Theorem**; premises measured | Appendix B.5.3; tail energy $12$–$16\%$ ($\varepsilon_{\mathrm{tail}}\approx0.35$–$0.40$), mismatch exponent $0.99$ | sites satisfying closure and tail bounds |',
    r'| Thm. 5 | carrier sufficiency: three-term $L^2$ bound on read-in error, action mismatch and target tail | **Theorem** (an identity) | Appendix B.5.3; the measured conditions size its terms: tail energy $12$–$16\%$ ($\varepsilon_{\mathrm{tail}}\approx0.35$–$0.40$), mismatch exponent $0.99$ | any carrier, action and read-in; instantiated for colour |')

lit('fix_appB_status_thm5',
    r'| Theorem 5 (carrier sufficiency, three-term bound) | Proved; premises measured | sites satisfying closure and tail bounds | B.5.3 |',
    r'| Theorem 5 (carrier sufficiency, three-term bound) | Proved (identity) | any carrier, action and read-in | B.5.3 |')

lit('fix_thm4_glance',
    r'| Chain error obeys a three-regime accumulation law | hue chains flat; heat chains degrade within the contractive bound ($0.760\to0.481$) | any realised family |',
    r"| Chain error obeys a three-regime accumulation law | hue chains flat; heat chains degrade monotonically ($0.760\to0.481$); the regime at a site is measured in the consumer's seminorm (Table D.22) | any realised family |")

lit('fix_thm4_claimscope',
    r'| Accumulation law (three-regime bound, Thm 4) | Any realised family, truth action with Lipschitz constant L | §5.4 (proof; the contractive and isometric regimes are measured) |',
    r"| Accumulation law (three-regime bound, Thm 4) | Any realised family, truth action with Lipschitz constant L | §5.4 (proof; the regime at each site is measured in the consumer's seminorm, Table D.22) |")

lit('fix_thm4_appA',
    r'| Thm. 4 | accumulation law $E_n\le\varepsilon+LE_{n-1}$; bounded / linear / exponential by $L\lessgtr1$ | **Theorem**; regimes verified | Appendix B.5.2; hue chains flat, heat chains monotone decreasing (Table 5) | any realised family, uniform single-step bound |',
    r"| Thm. 4 | accumulation law $E_n\le\varepsilon+LE_{n-1}$; bounded / linear / exponential by $L\lessgtr1$ | **Theorem**; the regime at a site is measured in the consumer's seminorm (§5.4, Table D.22) | Appendix B.5.2; hue chains flat, heat chains monotone decreasing (Table 5) | any realised family, uniform single-step bound |")

lit('fix_thm4_appB_status',
    r'| Theorem 4 (accumulation law) | Proved; qualitative regimes verified | any realised family, uniform single-step bound | B.5.2 |',
    r"| Theorem 4 (accumulation law) | Proved; the regime at a site is measured in the consumer's seminorm | any realised family, uniform single-step bound | B.5.2 |")

lit('fix_thm4_uniformity',
    r'Estimating $\varepsilon$ and $L$ independently of the measured chain error, rather than fitting the recursion to it, is not done here.',
    r"Estimating $\varepsilon$ and $L$ independently of the measured chain error, rather than fitting the recursion to it, is not done here; which of the three regimes a site occupies is measured in the consumer's seminorm (Table D.22), not inferred from the generator.")

lit('fix_propAB_2_1',
    'Our Propositions A–B supply the missing condition',
    'Our fibre condition supplies the missing information condition')

lit('fix_propAB_heading_B2',
    '#### B.2 The general layer (Propositions A and B)',
    '#### B.2 The general layer (the fibre condition)')

lit('fix_map_T3',
    r'| T3 | Retained rank is nearly irrelevant; mixing is decisive | the law contains κ alone | §3.4 (rank sweep vs κ sweep) |',
    r'| T3 | Retained rank is nearly irrelevant; mixing is decisive | the law contains $\kappa_W$ alone; rank enters only through where the split is drawn | §3.4 (rank sweep vs κ sweep) |')

lit('fix_Bdelta',
    r'the residual is $\|BA_\Delta\Sigma_c^{1/2}P_{\ker(B_\Delta\Sigma_c^{1/2})}\|_F^2$',
    r'the residual is $\|BA_\Delta\Sigma_c^{1/2}P_{\ker(B\Sigma_c^{1/2})}\|_F^2$')

lit('fix_crossing_8_4',
    'the four-block split, the crossing term — while the score is how we see it.',
    'the four-block split, the gate-change term — while the score is how we see it.')

lit('fix_crossing_consequence',
    'Measured (ResNet-50, three depths × two algebras): the flip count does not order the score ($\\rho=+0.49$, not significant), the crossing share does ($\\rho=-0.83$)',
    'Measured (ResNet-50, three depths × two algebras): the flip count does not order the score ($\\rho=+0.49$, not significant), the gate-change share — the share of the change carried by $\\Delta_{\\mathrm{flip}}$ — does ($\\rho=-0.83$)')

lit('fix_chains_5_4',
    'chains reproduce a direct fit for both algebras while the single-step substitute does not',
    'chains come within a few points of a direct fit for both algebras while the single-step substitute does not')

lit('fix_7_3_ref',
    'the dissipative family fails the closure premise in every basis tested (§4.7)',
    'the dissipative family has no shared decay rate in any basis tested (§8.3, Table 1)')

lit('fix_B511_heading',
    '##### B.5.1 Capacity boundary of per-dimension families',
    '##### B.5.1 Coordinate-monotone families on a closed orbit')

lit('fix_six_sites',
    'Six further sites in our real-network grid are causally unreachable in the same sense',
    'Two further sites — six site×parameter points — in our real-network grid are causally unreachable in the same sense')

lit('fix_principal_angle',
    'with rotation planes shared across shapes (principal angles within $0.96$ of identity)',
    'with rotation planes shared across shapes (principal-angle cosine at least $0.96$)')

lit('fix_A2_blank',
    '\n\n| The sharing term of the two-source decomposition is an in-split artefact',
    '\n| The sharing term of the two-source decomposition is an in-split artefact')

# ---- companion-document pointers (the paper must not point at another submission line) ----
lit('fix_src_C4',
    '`results/depth_map_<backbone>_{hue,heat}.json`, `paper/targets/TPAMI/DEPTH_MAP_TABLES.md`, `results/heat_semigroup_pilot.json`.',
    '`results/depth_map_<backbone>_{hue,heat}.json`, `results/heat_semigroup_pilot.json`.')

lit('fix_src_D13',
    "*Source: `results/multiattr_summary.json` (`synthetic_vitb16`); perimeter row of §4.7 and this project's supplement (`paper/targets/TPAMI/SUPPLEMENT.md`), Table S15.*",
    "*Source: `results/multiattr_summary.json` (`synthetic_vitb16`); perimeter row of §4.7.*")

lit('fix_src_D14',
    'the table is rewritten from that summary by `scripts/252_update_real_multiattr_table.py`. This project\'s supplement (`paper/targets/TPAMI/SUPPLEMENT.md`), Table S16, carries the superseded numbers.*',
    'the table is rewritten from that summary by `scripts/252_update_real_multiattr_table.py`.*')

lit('fix_caption_F1_a',
    'so the five-arm stage-1 comparison is not at a matched fitting budget for the plain-CNN row (the disclosed inconsistency in this project\'s supplement (`paper/targets/TPAMI/SUPPLEMENT.md`), Table S11 covers the stage-2 part only).',
    'so the five-arm stage-1 comparison is not at a matched fitting budget for the plain-CNN row (the stage-2 part of the same disclosure is in the per-arm files named below).')

lit('fix_src_F2_table',
    "*Source: this project's supplement (`paper/targets/TPAMI/SUPPLEMENT.md`), Table S1 — no raw `results/` file holds this five-arm benchmark.*",
    "*Source: `results/c10_routeB/summary.json` and the per-(arm, seed) files `results/c10_routeB/<arm>_s<seed>.json`; produced by `scripts/c10_routeB_train.py`, `scripts/c10_routeB_full.py`, `scripts/c10_routeB_summary.py`. OOD sweep: 37 hue-shift factors in $[-0.5, 0.5]$ of the full circle.*")

lit('fix_note_F2',
    "not of the interface. *Source: `paper/targets/TPAMI/SUPPLEMENT.md`, Table S1 (five-arm benchmark).*",
    "not of the interface. *Source: `results/c10_routeB/summary.json`.*")

lit('fix_src_F3',
    '`results/vit_causal_dinov2b14_b4_s{0,1,2}.json`; reproduces this project\'s supplement (`paper/targets/TPAMI/SUPPLEMENT.md`), Table S14; produced by scripts 155, 157.*',
    '`results/vit_causal_dinov2b14_b4_s{0,1,2}.json`; produced by scripts 155, 157.*')

lit('fix_src_F9',
    '(`--threads 4`, `DEV = "cpu"`); `paper/targets/TPAMI/EXPERIMENTS_SUMMARY.md` §3.7 |',
    '(`--threads 4`, `DEV = "cpu"`) |')

lit('fix_src_F8',
    "The COCO read-out cell quotes the stratified file ($8.0°$ high-concentration, $10.0°$ non-circular); an earlier draft's $9.0°$ came from the superseded `results/usage_rule_scale.json` and the smaller 12-region probe reads $9.25°$ (`results/region_rho_demo.json`) — three different quantities, all named in Appendix C.7; produced by scripts 109, 156, 231, 67.*",
    "The COCO read-out cell quotes the stratified file ($8.0°$ high-concentration, $10.0°$ non-circular), while the smaller 12-region probe reads $9.25°$ (`results/region_rho_demo.json`) — different quantities, all named in Appendix C.7; produced by scripts 109, 156, 231, 67.*")

lit('fix_boxE4',
    'the device string is recorded with the timing artefact, and the machine is documented as running at 3.3 GB used and 92% utilisation when the later verification runs were launched. The later verification runs are CPU-only, launched as `OMP_NUM_THREADS=4 nice -n 15`',
    'the device string is recorded with the timing artefact. The later verification runs are CPU-only, launched as `OMP_NUM_THREADS=4 nice -n 15`')

span('fix_appendix_intro',
     '- **Appendix A.** Claim status of every numbered and headline claim',
     '`paper/v15_appendices/G_positioning.md`',
     'The appendices follow in that order.')

# ---- cross-reference fixes ----
lit('fix_7_1_tableref',
    'Each measured fact specifies a design choice (Table 4):',
    'Each measured fact specifies a design choice (Table 12):')

lit('fix_7_6_tableref',
    'composition reaches parameters never fitted (Table 5);',
    'composition reaches parameters never fitted (Table D.11);')

lit('fix_5_5_secref',
    'The random orthogonal operator of §1.2 attains the best projection score at $15$ of $42$ sites',
    'The random orthogonal operator of §1.3 attains the best projection score at $15$ of $42$ sites')

# ---- numbers that lacked a pointer ----
span('fix_087_main',
     '*Consequence.* **Even with no gate change at all',
     'agree to within $0.087$ at every site (§5.2).',
     r"""*Consequence.* **Even with no gate change at all, the multi-region structure alone can obstruct a fixed linear action**: the visited regions must be simultaneously conjugate to the transformation. This is why enlarging the operator family buys little — measured, a ridge-plus-$3\times3$-residual family differs from the global linear family by at most $0.087$ at every ResNet-50 site and $0.11$ at every ConvNeXt site, with a median gap of $0.004$ over the 32-site grid; the two separate only at the deepest class-token blocks, where both fall below zero. *Source: `results/depth_map_summary.json`.*""")

span('fix_087_appB',
     '*Consequence.* **Even with no crossing at all',
     'agree to within $0.087$ at every site (§5.2).',
     r"""*Consequence.* **Even with no crossing at all, the multi-region structure alone can obstruct a global linear action**: the visited regions must be simultaneously conjugate to the transformation. This is why enlarging the operator family buys little — measured, a ridge-plus-$3\times3$-residual family differs from the global linear family by at most $0.087$ at every ResNet-50 site and $0.11$ at every ConvNeXt site, with a median gap of $0.004$ over the 32-site grid; the two separate only at the deepest class-token blocks, where both fall below zero. *Source: `results/depth_map_summary.json`.*""")

lit('fix_6_2_source',
    'The stronger reading of the same evidence — a symbolisation transition from continuous fields to discrete clusters — remains a hypothesis: supported by the geometry, not isolated from the coverage growth that accompanies it.',
    'The stronger reading of the same evidence — a symbolisation transition from continuous fields to discrete clusters — remains a hypothesis: supported by the geometry, not isolated from the coverage growth that accompanies it. *Source: `results/field_measure_resnet18_l4.json`, `results/field_measure_yolo11n_y7.json` (orbit bend and closed-loop error per (shape, saturation, value)).*')

lit('fix_8_3_source',
    'the canonical spatial action does not compile, and the family is carried in channel space instead.',
    'the canonical spatial action does not compile, and the family is carried in channel space instead. *Source: `results/heat_semigroup_structure.json`.*')

lit('fix_table10_setup',
    'unseen shapes, 64-cell grid; 120 real regions |',
    'unseen shapes, 64-cell grid; 120 real regions (69 high-concentration) |')

lit('fix_table10_source_add',
    '`results/region_rho_demo.json`, `results/code3d_sv_lowext.json`.; produced by scripts 67, 231, 176, 156, 109, 112.*',
    '`results/region_rho_demo.json`, `results/code3d_sv_lowext.json`, `results/usage_rule_scale_strat.json` (real-region read-out); produced by scripts 67, 231, 176, 156, 109, 112.*')

lit('fix_uniqueness',
    'When a realising operator exists it is unique; when it does not,',
    'When a realising operator exists it is unique, being the solution of $\\rho W=Wg$ with $W$ of full row rank; when it does not,')

lit('fix_6_5_diagnostic',
    'the signed deviation locates which premise fails first — the exact law remains the theory\'s linear layer, its breakdown pattern is diagnostic, and the structure–operability links above are measured, not extrapolated.',
    'the exact law remains the theory\'s linear layer, its breakdown pattern is diagnostic rather than predictive — §3.4 reports that its ordering is no better than the naive baselines — and the structure–operability links above are measured, not extrapolated.')

lit('fix_appA_governs',
    "| The crossing's concentration, not the flip count, governs | Measured | concentration $\\rho=-0.83$ ($p=0.042$); consumer-visible share $\\rho=-0.886$ ($p=0.019$); flip count $\\rho=+0.49$ n.s. | ResNet-50, $n=6$ |",
    "| The gate-change concentration, not the flip count, orders the measured transfer | Measured | gate-change share $\\rho=-0.83$ ($p=0.042$); consumer-visible share $\\rho=-0.886$ ($p=0.019$); flip count $\\rho=+0.49$ n.s. | ResNet-50, $n=6$ |")

# ---- overclaims and process notes ----
lit('fix_sec9_composite',
    'competing an unfitted composite is not harder than refitting it,',
    'on the measured arms a composite fitted only at its parts is not worse than a direct fit (O8 $0.844$ against $0.807$ at $\\sigma=1.803$, seed 0; the three-seed fixed-total chains of Table 5 are $0.451$–$0.488$ against a direct $0.511$),')

lit('fix_5_3_omission',
    'Two further gaps — the median advantage of a local step over a global single direction at a fixed site, and how that gap grows with depth — were reported in earlier drafts; we no longer report them, because we cannot trace them to a producing artifact and therefore cannot stand behind the numbers.',
    'Two further gaps — the median advantage of a local step over a global single direction at a fixed site, and how that gap grows with depth — are omitted: no producing artefact exists for them in this project, and a number without one is not reported here.')

lit('fix_B_intro_meta',
    'where a statement is not proved, its status is said in the same place rather than left to be inferred. **Nothing here changes a claim, a number or a scope.** The summary of statuses is Table B.4.',
    'where a statement is not proved, its status is said in the same place rather than left to be inferred. The summary of statuses is Table B.4.')

# ---- gaps: tension specifics, and the design condition's reach ----
lit('fix_4_6_tension_specifics',
    'The two differ in model, data, conditioning information and capacity, and one conclusion follows from them jointly:',
    'The two differ in model, data, conditioning information and capacity: the content-conditioning contrasts fit a content-conditioned operator family at matched capacity on the arms they name, on synthetic orbits and COCO regions (Table 3), whereas the cross-content statement here concerns a global linear map on a per-site carrier synthesis estimated from dense, controlled orbits. One conclusion follows from them jointly:')

lit('fix_4_6_compiled_carrier',
    'it says which coordinate sets a compiled carrier may keep, and what a chosen truncation costs.',
    'it says which coordinate sets a compiled carrier may keep, and what a chosen truncation costs. The compiled interface of §7 keeps all six coordinates, so the condition is satisfied there by construction; what it constrains is any *reduced* carrier, and the dropped-coordinate variant of Table D.21 shows the price of a truncation that is not invariant under the action.')

# ---- direct test of the fibre condition (existing experiment, previously unreported) ----
span('add_star_main',
     '**The organising principle.**',
     '**The organising principle.**',
     r"""**A direct check of the fibre condition (★).** The condition can be tested without fitting any operator. Two inputs that a frozen consumer maps to the same output must, if the correspondence exists, stay equal after the same physical transformation. Fixing the tolerance at the $q$-quantile of the consumer's base pair-distance distribution — so that the random-pair preservation rate is $q$ by construction — the fraction of base-equal pairs that remain equal after the transformation is $0.25$–$0.46$ for hue and $0.58$–$0.82$ for heat at $q=0.01$, across ResNet-50, ConvNeXt-T, ViT-B/16 and DINOv2-B/14, far above the $0.01$ chance rate and with the same ordering at $q=0.05$. Existence therefore holds approximately rather than exactly, and more strongly for the dissipative family, which merges states instead of moving them. The full grid, including the per-site rates and the effect sizes, is Table D.23. *Source: `scripts/229_star_existence_test.py`; `results/star_existence_test.json` (300 COCO crops, CPU-only, seed 0).*

**The organising principle.**""")

span('add_star_appendix',
     '---\n\n### Appendix E: Method boxes',
     '---\n\n### Appendix E: Method boxes',
     r"""#### D.9 Direct test of the fibre condition

**Table D.23.** The fibre condition tested directly at the consumer and at each site: the fraction of base-equal pairs that remain equal after the transformation, at tolerance quantiles $q=0.01$ and $q=0.05$ of the base pair-distance distribution (the random-pair baseline is $q$ by construction), with the transformation's effect size on the consumer and the per-site rates at $q=0.01$. Shallow to deep, left to right.

| backbone | family | effect size | consumer $q=0.01$ | consumer $q=0.05$ | per-site $q=0.01$ (four sites, shallow$\to$deep) |
|---|---|---|---|---|---|
| ResNet-50 | hue | 0.724 | 0.327 | 0.499 | 0.59 / 0.67 / 0.63 / 0.31 |
| ResNet-50 | heat | 0.594 | 0.595 | 0.655 | 0.61 / 0.57 / 0.51 / 0.61 |
| ConvNeXt-T | hue | 0.708 | 0.454 | 0.599 | 0.55 / 0.56 / 0.31 / 0.45 |
| ConvNeXt-T | heat | 0.517 | 0.577 | 0.612 | 0.55 / 0.51 / 0.35 / 0.18 |
| ViT-B/16 | hue | 0.812 | 0.249 | 0.332 | 0.67 / 0.70 / 0.59 / 0.38 |
| ViT-B/16 | heat | 0.465 | 0.710 | 0.727 | 0.62 / 0.62 / 0.70 / 0.56 |
| DINOv2-B/14 | hue | 0.659 | 0.463 | 0.609 | 0.74 / 0.76 / 0.69 / 0.57 |
| DINOv2-B/14 | heat | 0.262 | 0.815 | 0.879 | 0.75 / 0.80 / 0.75 / 0.82 |

*Source: `results/star_existence_test.json`, produced by `scripts/229_star_existence_test.py` (300 COCO crops, seed 0, CPU-only; the tolerance is a quantile of the base pair-distance distribution, so the random-pair preservation rate is $q$ by construction and the reported rate is directly comparable to it; effect size is the relative change of the logits under the real transformation, which gates whether the test has power at all).*

---

### Appendix E: Method boxes""")


# ---- registered prose-number target: keep the main text at or below 450 ----
lit('trim_contrib1_sections',
    '1. **Theory: when a transformation law is realisable at all** (§3, §4.6, §5.2, §5.4, §7.3). A realisability theory in three layers.',
    '1. **Theory: when a transformation law is realisable at all.** A realisability theory in three layers.')

lit('trim_sec9_composite',
    "on the measured arms a composite fitted only at its parts is not worse than a direct fit (O8 $0.844$ against $0.807$ at $\\sigma=1.803$, seed 0; the three-seed fixed-total chains of Table 5 are $0.451$–$0.488$ against a direct $0.511$),",
    "on the measured arms a composite fitted only at its parts is not worse than a direct fit (Table D.11), while the three-seed fixed-total chains fall a few points short of a direct refit (Table 5),")

lit('trim_star_backbones',
    'across ResNet-50, ConvNeXt-T, ViT-B/16 and DINOv2-B/14, far above the $0.01$ chance rate and with the same ordering at $q=0.05$.',
    'across four frozen backbones, far above the chance rate that the tolerance quantile fixes by construction.')

lit('trim_4_1_backbones',
    'passed through frozen backbones (ResNet-18/50, ConvNeXt, ViT-B/16, DINOv2-B/14 (Oquab et al., TMLR 2024), and a YOLO',
    'passed through frozen backbones (four architectures, one of them self-supervised (Oquab et al., TMLR 2024), and a YOLO')

lit('trim_3_5_gate',
    ' and on each region the site map is affine, $\\phi(x)=A_Gx+b_G$. The gate is recoverable from the site tensor itself ($z=\\mathrm{ReLU}(a)$, so $G=\\mathbb 1[z>0]$). Define',
    ' and on each region the site map is affine, $\\phi(x)=A_Gx+b_G$. Define')

lit('trim_6_5_numbers',
    'Where the linear law is extrapolated to real nonlinear sites, its closed form deviates systematically (+0.40 to +0.65 across 42 sites). We treat this as',
    'Where the linear law is extrapolated to real nonlinear sites, its closed form deviates systematically. We treat this as')

lit('fix_src_F2_paths',
    '`results/c10_routeB/summary.json` and the per-(arm, seed) files `results/c10_routeB/<arm>_s<seed>.json`; produced by',
    '`results/c10_routeB/summary.json` and the per-(arm, seed) files in `results/c10_routeB/`, e.g. `results/c10_routeB/z2_s0.json`; produced by')


lit('fix_src_D_depth_tables',
    '*Source: `results/depth_map_summary.json`; per-site tables and the $\\sigma=1$ single-seed grid in `DEPTH_MAP_TABLES.md`; produced by scripts 188, 190.*',
    '*Source: `results/depth_map_summary.json` and the per-site `results/depth_map_<backbone>_{hue,heat}.json`; produced by scripts 188, 190.*')

lit('fix_D_intro_reproduces',
    'Reproduces `DEPTH_MAP_TABLES.md` and Table 6; no value disagrees with either.',
    'Reproduces Table 6; no value disagrees with it.')

lit('fix_C7_superseded',
    'and `results/usage_rule_scale.json`, whose high-concentration subset also read $9.0°$, is superseded by the stratified file.',
    'and `results/usage_rule_scale.json` records the same protocol on a different region set.')


lit('fix_appA_propE_label',
    r"| Prop. E | $T_F\le1-\lVert\mathrm{Proj}_F\Delta_{\mathrm{flip}}\rVert_F/\lVert\mathrm{Proj}_F(z'-z)\rVert_F$ | **Empirical bound** (exact form proved for linear consumers) | 6/6 site–algebra points; slack = within-regime non-linearity | fixed-linear class |",
    r'| Prop. E | $T_F^{\max}=1-\lVert\mathrm{Proj}_F d^\perp\rVert_F/\lVert\mathrm{Proj}_F d\rVert_F$ for the fixed-linear class | **Theorem** for linear consumers, verified at 6/6 site–algebra points; the gate-change share is an empirical proxy for its slack | 6/6 site–algebra points | fixed-linear class |')

lit('fix_F2_caption_thm2',
    r'(B) The defect is the mixing ratio $\kappa$: measured $1-T_{RMS}$ against the closed form $\kappa/\sqrt{1+\kappa^2}$ over the sweep, maximum deviation $0.0091$',
    r'(B) The defect is the mixing ratio in the metric the encoder induces: measured $1-T_{RMS}$ against the closed form of Theorem 2 over the sweep, which is run in the controlled isotropic setting where $\kappa_W=\kappa$, with maximum deviation $0.0091$')

lit('fix_B2_approx_price',
    'the approximate form is what the local transporter of §5.3 accepts deliberately, at the price priced by Theorem 4.',
    'the approximate form is what the local transporter of §5.3 accepts deliberately, and the composition price it pays is the one §5.4 measures.')

lit('fix_F8_routes',
    '(B) Route by route: $69.5$, $12.6$ and $10.0°$ median.',
    '(B) Route by route: $69.5°$ raw, $12.6°$ dominant-fill and $10.0°$ shifted-fill, median.')

lit('fix_F8_cost',
    '(E) Cost: $66$ ns per box against $38$ ms for a re-forward.',
    '(E) Cost: a code rotation costs nanoseconds per object against $38$ ms for a re-forward and re-colour (Table E1).')


def lit_all(name, old, new):
    OPS.append(('litall', name, old, None, new))


# ---- second verification round ----
lit_all('fix2_glance_bound_row',
        r'| Achievable faithfulness is bounded by the consumer-visible crossing share, not by the flip count | $6/6$ site–algebra points satisfy the bound; $\rho=-0.886$ visible against $+0.49$ (n.s.) for flip count | piecewise-linear encoders, measured |',
        r'| The gate-change concentration orders the measured transfer, and a change of local rule counts only where the consumer reads it | $\rho=-0.886$ for the consumer-visible share against $+0.49$ (n.s.) for the flip count | piecewise-linear encoders, measured |')

lit_all('fix2_map_T7',
        r'| T7 | Achievable faithfulness is bounded by the consumer-visible share of the change | Prop E, for the fixed-linear class | §3.5 (proof sketch + 6/6 empirical) |',
        r'| T7 | The fixed-linear class attains the exact score $T_F^{\max}$, and the gate-change share is a measured proxy for its slack | Prop E | §3.5 (proof + six-point check) |')

lit_all('fix2_contrib_propE',
        'for the fixed-linear class achievable faithfulness is bounded by the consumer-visible share of the change (Proposition E)',
        'for the fixed-linear class the attainable score has an exact form, with the gate-change share as a measured proxy for its slack (Proposition E)')

lit_all('fix2_propE_status_main',
        '*Status.* This exact form is proved for linear consumers and verified at six (site, algebra) points; no claim is made here for families outside the fixed-linear class.',
        '*Status.* This exact form is proved for linear consumers; no claim is made here for families outside the fixed-linear class. The proxy is evaluated against the measured transfer at six (site, algebra) points in §6.3.')

lit_all('fix2_propE_status_app',
        '*Status.* Proved for linear consumers, verified at six (site, algebra) points. No claim is made for families outside the fixed-linear class.',
        '*Status.* Proved for linear consumers. No claim is made for families outside the fixed-linear class; the proxy is evaluated at six (site, algebra) points in §6.3.')

lit_all('fix2_appA_propE_6of6',
        r'| Prop. E | $T_F^{\max}=1-\lVert\mathrm{Proj}_F d^\perp\rVert_F/\lVert\mathrm{Proj}_F d\rVert_F$ for the fixed-linear class | **Theorem** for linear consumers, verified at 6/6 site–algebra points; the gate-change share is an empirical proxy for its slack | 6/6 site–algebra points | fixed-linear class |',
        r'| Prop. E | $T_F^{\max}=1-\lVert\mathrm{Proj}_F d^\perp\rVert_F/\lVert\mathrm{Proj}_F d\rVert_F$ for the fixed-linear class | **Theorem** for linear consumers; the gate-change share is an empirical proxy for its slack | evaluated against the measured transfer at six site–algebra points (§6.3) | fixed-linear class |')

lit_all('fix2_thm5_conditional',
        'The theorem is conditional, its conditions are measurable, and the ordering it predicts — carrier error tracking the tail and the plane misalignment site by site — is not tested here; the bound is sufficient rather than tight.',
        'The three-term inequality is sufficient rather than tight, and the ordering it predicts — carrier error tracking the tail and the plane misalignment site by site — is not tested here.')

lit_all('fix2_thm5_general_8_2',
        'and carrier sufficiency (Thm 5) is stated for any site meeting its premises, which is why it can return a *negative* answer for an algebra',
        'and carrier sufficiency (Thm 5) holds as an identity for any carrier, action and read-in, with the measured conditions sizing its terms, which is why it can return a *negative* answer for an algebra')

lit_all('fix2_claimscope_thm5',
        r'| Carrier sufficiency (Thm 5) and the interface | Any site meeting the measured premises; instantiated for colour | §7.3 (proof + measured premises) |',
        r'| Carrier sufficiency (Thm 5) and the interface | Any carrier, action and read-in; the measured conditions size its terms, and it is instantiated for colour | §7.3 (proof + measured conditions) |')

lit_all('fix2_4_3_conditions',
        "These two measurements are the theory's premises arriving as numbers",
        "These two measurements are the construction's conditions arriving as numbers")

lit_all('fix2_boxE2_licence',
        "the licence for the fixed action is Theorem 5's premises, which are measured.",
        'the licence for the fixed action is that the two conditions the construction uses are measured to hold at the site (§4.3, §7.3).')

lit_all('fix2_a2_chains_row',
        '| Chains reproduce a direct fit; single-step substitutes do not |',
        '| Chains come within a few points of a direct fit; single-step substitutes do not |')

lit_all('fix2_grid_scope_crossing',
        'the crossing-term and semigroup-structure studies',
        'the gate-change and semigroup-structure studies')

lit_all('fix2_8_5_crossing',
        'The crossing-term study, the semigroup-structure study and the mild-dissipation grid are single-seed',
        'The gate-change study, the semigroup-structure study and the mild-dissipation grid are single-seed')

lit_all('fix2_D7_crossing',
        'The depth grid and the crossing diagnosis agree.',
        'The depth grid and the gate-change diagnosis agree.')

lit_all('fix2_6_3_crossing_terms',
        'the consumer-visible share that orders the transfer score ($\\rho=-0.886$), more tightly than the raw feature-level crossing share ($\\rho=-0.83$)',
        'the consumer-visible share of the change orders the transfer score ($\\rho=-0.886$), more tightly than the raw gate-change share ($\\rho=-0.83$)')

lit_all('fix2_8_3_dispersion_order',
        'the rate dispersion is $8.6$ and $17.5$ at layers 2 and 3 against $0.91$',
        'the rate dispersion is $17.5$ and $8.6$ at layers 2 and 3 against $0.91$')

lit_all('fix2_6_2_numbers',
        'orbit bending rises to 725–729° at the deepest measured layers, closed-loop error worsens (up to 0.16 normalised)',
        'orbit bending rises to a mean of 729° at the deepest measured layer (up to 760° in a single (shape, saturation, value) cell), closed-loop error worsens (up to 0.24 normalised in the same file)')

lit_all('fix2_087_median',
        'with a median gap of $0.004$ over the 32-site grid; the two separate only at the deepest class-token blocks, where both fall below zero.',
        'with a median gap of $0.005$ over the 32-site grid; the two separate only where both fail (DINOv2 heat $block8$: $-0.40$ against $-0.99$).')

lit_all('fix2_star_F',
        '**A direct check of the fibre condition (★).**',
        '**A direct check of the fibre condition (F).**')

lit_all('fix2_D9_sd',
        '| $45+45$ | 0.237 ± 0.013 | 0.472 ± 0.024 |',
        '| $45+45$ | 0.237 ± 0.013 | 0.472 ± 0.023 |')

lit_all('fix2_src_D15',
        "*Source: `results/second_attribute_chain.json` (`identify`, `geometry`, `spatial_fallback`); Supplement §I.2; produced by scripts 176.*",
        "*Source: `results/second_attribute_chain.json` (`identify`, `geometry`, `spatial_fallback`); produced by scripts 176.*")

lit_all('fix2_src_F9_memo',
        "*Source: `results/perf_reforward_vs_rho.json` (`device`), `results/region_rho_demo.json`; `paper/TPAMI_READINESS_ASSESSMENT.md` (GPU occupancy and the CPU-only `OMP_NUM_THREADS=4 nice -n 15` policy).*",
        "*Source: `results/perf_reforward_vs_rho.json` (`device`), `results/region_rho_demo.json`.*")

lit_all('fix2_5_3_omission_reword',
        'are omitted: no producing artefact exists for them in this project, and a number without one is not reported here.',
        'are not reported here, because no producing artefact exists for them.')


lit_all('fix2_thm3_proof_pointer',
        r'contributes $L^2$ error at least $A^2\ell/4$.',
        r'contributes $L^2$ error at least $A^2\ell/4$. (Proved in Appendix B.5.1; the two lemmas are B.7 and B.8.)')

lit_all('fix2_corC1_proof_pointer',
        r"$\rho b_r=b_r$ for every visited $r$. For a single visited region this is Theorem 1's kernel condition rewritten — the linear layer is the single-region case.",
        r"""$\rho b_r=b_r$ for every visited $r$. *Proof.* With the gate unchanged the realised transitions are the diagonal cells, so the joint system of Proposition C reduces to its diagonal part and nothing else is imposed. $\square$

For a single visited region this is Theorem 1's kernel condition rewritten — the linear layer is the single-region case.""")


lit_all('fix3_claimscope_propE',
        r'| Region-transition criterion (Prop C, Cor. C.1), gate-change decomposition (Prop D), achievable-score bound (Prop E) | **Piecewise-linear encoders**; E for the fixed-linear class, measured | §3.5 (proofs; E empirical) |',
        r'| Region-transition criterion (Prop C, Cor. C.1), gate-change decomposition (Prop D), exact score of the fixed-linear class (Prop E) | **Piecewise-linear encoders**; E for the fixed-linear class, with its proxy checked in §6.3 | §3.5 (proofs) |')

lit_all('fix3_appB_status_propE',
        r'| Proposition E (achievable-score bound) | Proved in exact form (linear consumers); **empirical** as displayed | fixed-linear class | B.4 |',
        r'| Proposition E (exact score of the fixed-linear class) | Proved for linear consumers; the gate-change share is an empirical proxy for its slack | fixed-linear class | B.4 |')

lit_all('fix3_appG_propE',
        'and bounded in the rectifier layer (Props C–E)',
        "and with the fixed-linear class's score exact in the rectifier layer (Props C–E)")

lit_all('fix3_F8_composition_row',
        r'| composition to a never-fitted parameter | $T_F$ at the composite | direct fit at that parameter | composed $\ge$ direct (Table D.11) |',
        r'| composition to a never-fitted parameter | $T_F$ at the composite | direct fit at that parameter | composed $0.844$ vs direct $0.807$ at $\sigma=1.803$ (Table D.11) |')

lit_all('fix3_6_3_grammar',
        'across the six (site, algebra) points it is the consumer-visible share of the change orders the transfer score',
        'across the six (site, algebra) points it is the consumer-visible share of the change that orders the transfer score')

lit_all('fix3_6_3_crossing_share',
        'the consumer-visible crossing share rises with depth under both algebras',
        'the consumer-visible gate-change share rises with depth under both algebras')

lit_all('fix3_slogan',
        '"A crossing counts only where the consumer reads it" is that statement',
        '"A gate change counts only where the consumer reads it" is that statement')

lit_all('fix3_slogan_8_4',
        'which is the same statement as §3.5\'s: a crossing counts only where the consumer reads it.',
        "which is the same statement as §3.5's: a gate change counts only where the consumer reads it.")

lit_all('fix3_8_3_conditional',
        'This is the shape of a conditional theory doing its job: carrier sufficiency (Theorem 5) has measurable premises, and on this instance it returns a negative answer for one of the two algebras while the truth-level law holds exactly.',
        'This is the shape of a measurable theory doing its job: the three-term bound of Theorem 5 has terms the measurements size, and on this instance the dissipative family fails them, so the construction returns a negative answer for one of the two algebras while the truth-level law holds exactly.')

lit_all('fix3_7_3_filter',
        'A carrier-sufficiency theorem with measurable premises is a filter rather than a promise',
        'A carrier-sufficiency statement whose terms are measurable is a filter rather than a promise')

lit_all('fix3_6_2_closure_source',
        'closed-loop error worsens (up to 0.24 normalised in the same file)',
        'closed-loop error worsens (up to 0.24 normalised at ResNet-18 layer 4)')

lit_all('fix3_D7_share_terms',
        'the raw feature crossing share, $-0.886$ for the consumer-visible crossing share',
        'the raw gate-change share, $-0.886$ for the consumer-visible gate-change share')

lit_all('fix3_appB_flip_share',
        r'against $\rho(\text{flip depth},T_F)=\rho(\text{crossing share},T_F)=-0.83$.',
        r'against $\rho(\text{flip depth},T_F)=\rho(\text{gate-change share},T_F)=-0.83$.')


lit_all('fix3_6_2_closure_source',
        'closed-loop error worsens (up to 0.24 normalised at ResNet-18 layer 4)',
        'closed-loop error worsens (up to 0.24 normalised)')

lit_all('fix3_trim_contrib3',
        '3. **Empirical law: the organisation of colour transformations and its erosion** (§4–§6).',
        '3. **Empirical law: the organisation of colour transformations and its erosion.**')


# ---- §7.2: the fixed action compared with learnable actions (script 265) ----
lit_all('add_7_2_comparison',
        'Two constructions in this paper are also distinct and stay distinct in the wording: the local transporter of §5.3 has its generator and Fourier coefficients fitted and then frozen at inference (Appendix E, Box E1), while the carrier rotation used here is never fitted at all (Box E2).',
        r"""Two constructions in this paper are also distinct and stay distinct in the wording: the local transporter of §5.3 has its generator and Fourier coefficients fitted and then frozen at inference (Appendix E, Box E1), while the carrier rotation used here is never fitted at all (Box E2).

**Compared with a learned action.** Whether the fixed action costs anything relative to learning it is now measured rather than left open. Under one read-in, one code supervision and one budget, four arms were trained: the fixed rotation, a learned generator, a learned linear action, and a generic conditioned map carrying more parameters than the fixed action has. The fixed action is exact in the increment it applies and in composition, and its read-out error is not worse than the others'; the learned generator comes closest, since one generator already composes exactly, the learned linear action fits the small increments it saw and fails on an unseen large one, and the generic map is worse than the fixed action on every action metric despite its extra capacity. Learning an action therefore buys nothing here that the measured form does not already give, and it can lose the two properties the construction exists to provide. Table F.10 carries all four arms, three seeds, the parameter counts and the protocol. *Source: `scripts/265_interface_action_comparison.py`; `results/interface_action_comparison.json`.*""")

lit_all('fix_7_2_tested',
        'whether a fitted action of equal capacity would do better is not tested here;',
        'whether a fitted action of equal capacity does better is measured below (Table F.10);')

lit_all('fix_boxE2_tested',
        'Whether a fitted action of equal capacity would do better is not tested here;',
        'Whether a fitted action of equal capacity does better is measured in §7.2 (Table F.10);')

span('add_table_F10',
     '---\n\n### Appendix G: Positioning against the four literature lines',
     '---\n\n### Appendix G: Positioning against the four literature lines',
     r"""**Table F.10.** The fixed, measured action against learnable actions under identical supervision. One read-in ($64\to256\to256\to6$), one code supervision, 1500 epochs $\times$ 24 minibatches $\times$ 96 code samples, train shapes 0–3 and unseen shapes 4–5; the three learnable arms additionally see paired increments drawn from the same hue labels. Errors are medians over the 226 unseen-shape cells, averaged over three seeds. *Increment* is the error of the increment the arm applies against the true increment, at $37°$ and $90°$, neither of them fitted; *composition defect* compares the two-step composite with the direct step.

| arm | action | action params | read-out | increment $37°$ | increment $90°$ | composition defect | composite vs truth |
|---|---|---|---|---|---|---|---|
| fixed (ours) | $A_\Delta=\operatorname{diag}(I_2,R(\Delta),R(2\Delta))$, never fitted | 0 | **1.45°** | **0.00°** | **0.00°** | **0.00°** | **0.00°** |
| learned generator | $\exp(\Delta G)$, $G\in\mathbb R^{6\times6}$ free | 36 | 2.10° | 1.31° | 1.02° | 0.00° | 1.02° |
| learned linear action | $I+\Delta B$, $B\in\mathbb R^{6\times6}$ free | 36 | 5.76° | 1.20° | 9.80° | 9.69° | 1.34° |
| generic conditioned map | $c+\mathrm{MLP}([c,\cos\Delta,\sin\Delta])$, 8-256-256-6 | 69,638 | 1.72° | 1.32° | 2.59° | 2.47° | 2.02° |

*Source: `scripts/265_interface_action_comparison.py`; `results/interface_action_comparison.json` (RTX 2060, three seeds; total parameters 83,974 / 84,010 / 84,010 / 153,612 in the rows' order). The fixed action's zeros are structural — a block rotation composes additively — and the learned generator's zero composition defect comes from using a single generator; the read-out column differs because each action is trained jointly with the same read-in.*

---

### Appendix G: Positioning against the four literature lines""")

lit_all('add_appA_measured_fixed_action',
        '*Source: statements and numbers as in the main text, each also traced in Tables B.1 and B.4.',
        r"""| The fixed action is not worse than learnable actions under identical supervision | Measured | fixed action exact in the increment and in composition; learned generator $1.0$–$1.3°$; learned linear action $9.8°$ at an unseen increment and $9.7°$ composition defect; generic conditioned map worse on every action metric at $1.8\times$ the parameters | synthetic orbits, unseen shapes, three seeds |

*Source: statements and numbers as in the main text, each also traced in Tables B.1 and B.4.""")




# ---- W1: figures 1-10 referenced from the text (each sentence describes what the figure shows) ----
lit_all('fix4_old_fig_ref', 'Figure F6', 'Figure 6')
lit_all('fig1_text',
        'confusing them is the root of the proxy failures in §5.5.',
        'confusing them is the root of the proxy failures in §5.5. Figure 1 fixes the two objects the paper compares: the action of a transformation on the world and the operator that stands in for it on features, and panel (b) shows which part of a linear encoder is responsible when the second cannot exist.')
lit_all('fig2_text',
        'the magnitude is a per-site quantity.',
        'the magnitude is a per-site quantity. Figure 2 shows the law in the controlled setting where it is exact, together with the two candidate explanations it rules out, retained rank and the amount of discarded energy.')
lit_all('fig3_text',
        'and it has **concrete degeneracies** that bound the operating domain (a $k=2$-only orbit cannot distinguish hues $180°$ apart; the grey axis is stabilised by every hue rotation).',
        'and it has **concrete degeneracies** that bound the operating domain (a $k=2$-only orbit cannot distinguish hues $180°$ apart; the grey axis is stabilised by every hue rotation). Figure 3 shows what that organisation consists of: two harmonics carry almost all of the energy, their planes are shared across shapes, and the low-saturation end of the domain is where the orbit weakens while keeping its plane.')
lit_all('fig4_text',
        'Section 6 picks up what training and depth then do to operability.',
        'Section 6 picks up what training and depth then do to operability. Figure 4 compares eight arms under one protocol and shows that the organisation is present before training, which is why the paper attributes it to the input statistics and the architecture rather than to learning.')
lit_all('fig5_text',
        'It is stated as a hypothesis, not a result.',
        'It is stated as a hypothesis, not a result. Figure 5 contrasts the two realisations site by site and prices the composition of a fixed total transformation.')
lit_all('fig6_text',
        '**Table 6** gives all eight curves.',
        '**Table 6** gives all eight curves. Figure 6 shows the depth profiles and the crossed design together: the curves decline, and the decline follows the pair formed by the transformation and the readout rather than the algebra.')
lit_all('fig7_text',
        "the applicability map they trace is part of the interface's specification, not an afterthought.",
        "the applicability map they trace is part of the interface's specification, not an afterthought. Figure 7 shows the interface as built: the readout error over the $(s,v)$ domain, the code magnitude that bounds it, the real-region comparison against the pre-registered threshold, and the cost of an edit.")
lit_all('fig8_text',
        'and region editing costs microseconds against milliseconds because it replaces the re-render it is compared with.',
        'and region editing costs microseconds against milliseconds because it replaces the re-render it is compared with. Figure 8 collects those uses on real images, each against its own baseline.')
lit_all('fig9_text',
        'The full grid, including the per-site rates and the effect sizes, is Table D.23.',
        'The full grid, including the per-site rates and the effect sizes, is Table D.23. Figure 9 shows the same check per backbone and per site.')
# --- the two figures added after the ten-figure set (W1.1): recovery, and out-of-sample prediction.
#     Both anchors are the sentences the ops above have just written, so they exist by construction and
#     are unique in the text at this point of the build.
# --- W2 (2026-09-16): the last internal traces, after the `*Source:` lines were moved to
#     meta/SOURCES.md.  Two appendix tables lose their "source" column, and six prose parentheticals
#     that named a result file or a script are dropped; the provenance lives in meta/SOURCES.md and
#     meta/ARTIFACTS.md.  Each edit asserts that its anchor occurs exactly once.
lit('w2_table7_header',
    r"""| quantity | value | source |
|---|---|---|
| local vs copy, per site (`locality_*`) | $1.55$–$15.94\times$ | `results/locality_*.json` |
| global vs copy, same sites | $0.08$–$0.84\times$ | `results/locality_*.json` |""",
    r"""| quantity | value |
|---|---|
| local vs copy, per site | $1.55$–$15.94\times$ |
| global vs copy, same sites | $0.08$–$0.84\times$ |""")
lit('w2_tableF9_header',
    r"""| phase | configuration | source |
|---|---|---|
| original training runs (CIFAR arms, depth grid, detector, interface) | single GPU, NVIDIA GeForce RTX 2060, 6 GB | `results/perf_reforward_vs_rho.json` (`device`); the 6 GB ceiling is stated in `scripts/188_depth_map_harness.py` |
| later verification runs (existence test and controls) | CPU-only, four threads (`OMP_NUM_THREADS=4`, `nice -n 15`), GPU deliberately untouched | `scripts/229_star_existence_test.py` (`--threads 4`, `DEV = "cpu"`) |""",
    r"""| phase | configuration |
|---|---|
| original training runs (CIFAR arms, depth grid, detector, interface) | single GPU, NVIDIA GeForce RTX 2060, 6 GB |
| later verification runs (existence test and controls) | CPU-only, four threads (`OMP_NUM_THREADS=4`, `nice -n 15`), GPU deliberately untouched |""")
lit('w2_probe_app', ' (`results/hue_probe_app.json`)', '')
lit('w2_proxy_bottom', ' (`results/transport_theory.json`, `proxy_bottom_variance`)', '')
lit('w2_deep_linear', ' (`results/deep_linear_transport.json`)', '')
lit('w2_carrier_opt', ' (`results/carrier_optimality_identity.json`, script 259)', '')
lit('w2_carrier_suff', ' (`results/carrier_sufficiency.json`, script 257)', '')
lit('w2_three_numbers_a',
    '`results/usage_rule_scale_strat.json` gives the high-concentration read-out',
    'the stratified high-concentration read-out, held out of the pre-registration,')
lit('w2_three_numbers_b',
    '`results/region_rho_demo.json` gives median $9.25°$ over twelve code-versus-real region-shift errors, the smaller earlier probe',
    'the smaller earlier probe gives median $9.25°$ over twelve code-versus-real region-shift errors')
lit('w2_three_numbers_c',
    'and `results/usage_rule_scale.json` records the same protocol on a different region set',
    'and a third run records the same protocol on a different region set')
lit('w2_boundary_indep',
    "The controlled-calibration AUC of $1.0$ is in `results/boundary_independent.json` (`auc_m_on_A`), and the main table's source line names it.",
    'The controlled-calibration AUC of $1.0$ is reported in Appendix C.')

# --- two §2 sentences that named a literature line without citing it (W5, added 2026-09-16)
lit('sec_2_2_reptheory',
    'Representation theory, in the classical sense, supplies the harmonic-block decomposition this paper uses as its candidate *forms*',
    'Classical representation theory [[cite:serre1977]] supplies the harmonic-block decomposition this paper uses as its candidate *forms*')
lit('sec_2_4_koopman',
    'Koopman-style approaches ask when a nonlinear system admits an invariant finite-dimensional subspace on which the evolution is linear',
    'Koopman-style approaches [[cite:lusch2018]] ask when a nonlinear system admits an invariant finite-dimensional subspace on which the evolution is linear')

# --- review fixes (2026-09-16): the report in paper/review/review.txt.  A1 (truncated appendix table),
#     A2 (the segment count), A3 (table numbers vs the order they appear), A4 (artifact names in an
#     appendix table), B1 (the ASCII commutation diagram), B2 (the contribution list numbering, fixed in
#     the converter), B3/B4 (every table captioned and numbered).
# A2: the hue circle is divided into nine segments of 40 degrees, not eight.
lit('review_a2_partition', 'The hue circle is divided into eight segments of $40°$',
    'The hue circle is divided into nine segments of $40°$')
lit('review_a2_generators', 'The eight per-segment generators are fitted by a third-order Fourier field',
    'The nine per-segment generators are fitted by a third-order Fourier field')
lit('review_a2_phases', 'least squares on the eight sampled phases',
    'least squares on the nine sampled phases')
lit('review_a2_budget', 'minibatch 64, eight segments', 'minibatch 64, nine segments')
# A3: a table is first cited in the section that discusses it, so the numbers follow the pages.  The
# three forward citations from the contributions list and from the closure discussion become section
# references, and the held-out decomposition table gains the citation it never had in the main text.
lit('review_a3_contrib',
    'measured on four pretrained backbones through four depths and two algebras (Table 6), with the price of a chain measured at a fixed total transformation (Table 5).',
    'measured on four pretrained backbones through four depths and two algebras (§6.1), with the price of a chain measured at a fixed total transformation (§5.4).')
lit('review_a3_closure',
    'including an oracle partition by the true displacement direction (Table 3)',
    'including an oracle partition by the true displacement direction (§5.2)')
lit('review_a3_heldout',
    'everything is then evaluated on a **held-out** split and normalised by the no-op displacement.',
    'everything is then evaluated on a **held-out** split and normalised by the no-op displacement. **Table 13** gives the held-out realisation error at each site and algebra.')
lit('review_a3_closure2',
    'on synthetic orbits and COCO regions (Table 3), whereas',
    'on synthetic orbits and COCO regions (§5.2), whereas')
lit('review_a3_protocol_row', 'erodes with depth (Table 6) | in-depth instance',
    'erodes with depth (§6.1) | in-depth instance')
lit('review_a3_capability',
    '**Table 10** carries both comparisons.',
    'The capability table of §7.6 carries both comparisons.')
# A4: Table B.1 loses the column that named the producing files and their keys.
lit('review_a4_header',
    '| manipulation | range | $T_F$ | closed-form deviation | source |\n|---|---|---|---|---|',
    '| manipulation | range | $T_F$ | closed-form deviation |\n|---|---|---|---|')
lit('review_a4_row1', ' | `transport_theory.json`, `lemma1_ker_preserving` |', ' |')
lit('review_a4_row2', ' | `transport_theory.json`, `theorem2_closed_form` |', ' |')
lit('review_a4_row3', ' | `transport_theory.json`, `corollary3_rank_sweep` |', ' |')
lit('review_a4_row4', ' | `transport_theory.json`, `proxy_bottom_variance` |', ' |')
lit('review_a4_row5', ' | `deep_linear_transport.json` |', ' |')
# B1: the ASCII commutation square is deleted; the same square is already drawn as Figure 1(a).
lit('review_b1_diagram', '```\n        τ ∈ M  (physical transformation: group / monoid / semigroup)\n   s ─────────────────►  τs\n   │                     │\n ψ_ℓ│                     │ ψ_ℓ\n   ▼                     ▼\n   z ─────────────────►  ρ_ℓ(τ) z ≈ ψ_ℓ(τs)\n        ρ_ℓ(τ): an operator on the feature space\n```',
    'that is, the square drawn in Figure 1(a).')
# B4: the three navigation tables of the main text become numbered tables with captions.  Numbers 14-16
# are placeholders above the current range; the renumbering pass assigns the final ones by citation.
lit('review_b4_glance',
    '**Main results at a glance.** Each row states the claim, its strongest evidence, and its scope.\n\n| Claim | Key evidence | Scope |',
    '**Table 14** collects the claims of this paper with the evidence that carries each one and its scope.\n\n**Table 14.** Main results at a glance: each row states the claim, the evidence that carries it, and its scope.\n\n| Claim | Key evidence | Scope |')
lit('review_b4_principle',
    "This single principle generates the paper's theory; the map below shows what it derives and where each consequence is earned:\n\n| # | Consequence | Derived how | Earned |",
    "This single principle generates the paper's theory, and **Table 15** lists what it derives and where each consequence is earned.\n\n**Table 15.** What the organising principle derives, and where each consequence is earned.\n\n| # | Consequence | Derived how | Earned |")
lit('review_b4_scope',
    '**Claim–scope table.** What is general and what is colour, fixed in one place:\n\n| Claim | Scope | Where earned |',
    '**Table 16** fixes what is general and what is colour in one place.\n\n**Table 16.** Claim–scope: what is general, what is specific to the colour instance, and where each is earned.\n\n| Claim | Scope | Where earned |')
# A1 + B4 for the appendix: the two claim-status tables are captioned and cited, and the long measured
# table is split into two so that neither half is taller than the page (A1: "Float too large for page").
lit('review_a1_caption',
    'No claim in the main text carries a stronger label than it carries here.\n\n#### A.1 Theory claims\n\n| # | claim | status | evidence | scope |',
    'No claim in the main text carries a stronger label than it carries here. **Table A.1**, **Table A.2** and **Table A.3** list every claim with the status it carries and the evidence that earns it.\n\n#### A.1 Theory claims\n\n**Table A.1.** Claim status: the theory claims, with the evidence that earns each one and the scope in which it holds.\n\n| # | claim | status | evidence | scope |')
lit('review_a2_caption',
    '#### A.2 Measured claims\n\n| claim | status | key numbers | scope |',
    '#### A.2 Measured claims: what the representation carries\n\n**Table A.2.** Claim status: the measured claims about the organisation of colour orbits and about which realisations work.\n\n| claim | status | key numbers | scope |')
lit('review_a3_split',
    '| Operability declines with depth | Measured |',
    '\n#### A.3 Measured claims: depth, learning and the interface\n\n**Table A.3.** Claim status: the measured claims about depth, learning, the two negative results and the interface.\n\n| claim | status | key numbers | scope |\n|---|---|---|---|\n| Operability declines with depth | Measured |')

# --- long display equations (2026-09-16): 13 of the 26 displays were wider than the 252pt column
#     and were being shrunk by \adjustbox to 0.63-0.87 of their size.  Each is now broken at a
#     relation or a sum into an aligned block, which keeps the type size; the generator still keeps
#     the max-width box as a safety net, and scripts/283_eqwidth_probe.py fails if one ever needs it
#     again (see meta/MANUSCRIPT_PLAN.md section 17).

lit('eqbreak__rho__ell_M_to_operatorname_',
    '$$\\rho_\\ell:M\\to\\operatorname{End}(Z_\\ell),\\qquad \\psi_\\ell(\\tau s)\\approx\\rho_\\ell(\\tau)\\psi_\\ell(s),\\qquad \\rho_\\ell(\\tau_2\\tau_1)=\\rho_\\ell(\\tau_2)\\circ\\rho_\\ell(\\tau_1),\\qquad \\rho_\\ell(e)=I,\\tag{★}$$',
    '$$\n\\begin{aligned}\n&\\rho_\\ell:M\\to\\operatorname{End}(Z_\\ell),\\qquad \\psi_\\ell(\\tau s)\\approx\\rho_\\ell(\\tau)\\psi_\\ell(s),\\\\\n&\\rho_\\ell(\\tau_2\\tau_1)=\\rho_\\ell(\\tau_2)\\circ\\rho_\\ell(\\tau_1),\\qquad \\rho_\\ell(e)=I,\n\\end{aligned}\\tag{★}\n$$')
lit('eqbreak__star_text_behavioural_',
    '$$(\\star\\text{-behavioural})\\qquad F(z_1)=F(z_2)\\ \\Longrightarrow\\ F(\\rho(\\tau)z_1)=F(\\rho(\\tau)z_2)\\quad\\text{for every }\\tau\\in M.$$',
    '$$\n\\begin{aligned}\n&(\\star\\text{-behavioural})\\qquad F(z_1)=F(z_2)\\\\\n&\\qquad\\Longrightarrow\\ F(\\rho(\\tau)z_1)=F(\\rho(\\tau)z_2)\\quad\\text{for every }\\tau\\in M.\n\\end{aligned}\n$$')
lit('eqbreak__text_within_cell_error_',
    '$$\\min_K\\ \\mathbb E\\|Ku-v\\|^2=\\underbrace{\\sum_{r,s}w_{rs}\\min_{K_{rs}}\\mathbb E\\big[\\|K_{rs}u-v\\|^2\\,\\big|\\,(r,s)\\big]}_{\\text{within-cell error}}+\\underbrace{\\sum_{r,s}w_{rs}\\,\\mathbb E\\big[\\|(K_{rs}-K^\\star)u\\|^2\\,\\big|\\,(r,s)\\big]}_{\\text{cost of sharing one operator}},$$',
    '$$\n\\begin{aligned}\n\\min_K\\ \\mathbb E\\|Ku-v\\|^2={}&\\underbrace{\\sum_{r,s}w_{rs}\\min_{K_{rs}}\\mathbb E\\big[\\|K_{rs}u-v\\|^2\\,\\big|\\,(r,s)\\big]}_{\\text{within-cell}}\\\\\n&+\\underbrace{\\sum_{r,s}w_{rs}\\,\\mathbb E\\big[\\|(K_{rs}-K^\\star)u\\|^2\\,\\big|\\,(r,s)\\big]}_{\\text{sharing cost}},\n\\end{aligned}\n$$')
lit('eqbreak_1_T__RMS_frac_lVert_WC_rVert',
    '$$1-T_{RMS}=\\frac{\\lVert WC\\rVert_F}{\\sqrt{\\lVert W(A-I_{\\mathcal R})\\rVert_F^2+\\lVert WC\\rVert_F^2}}\\;=\\;\\frac{\\kappa_W}{\\sqrt{1+\\kappa_W^2}},\\qquad \\kappa_W:=\\frac{\\lVert WC\\rVert_F}{\\lVert W(A-I_{\\mathcal R})\\rVert_F}.$$',
    '$$\n\\begin{aligned}\n1-T_{RMS}&=\\frac{\\lVert WC\\rVert_F}{\\sqrt{\\lVert W(A-I_{\\mathcal R})\\rVert_F^2+\\lVert WC\\rVert_F^2}}\n=\\frac{\\kappa_W}{\\sqrt{1+\\kappa_W^2}},\\\\\n\\kappa_W&:=\\frac{\\lVert WC\\rVert_F}{\\lVert W(A-I_{\\mathcal R})\\rVert_F}.\n\\end{aligned}\n$$')
lit('eqbreak_E_n_le_varepsilon_L_E__n_1_',
    '$$E_n\\;\\le\\;\\varepsilon+L\\,E_{n-1},\\qquad\\text{equivalently}\\qquad E_n\\le\\varepsilon\\frac{L^n-1}{L-1}\\ (L\\neq1),\\qquad E_n\\le n\\varepsilon\\ (L=1).$$',
    '$$\n\\begin{aligned}\nE_n&\\le\\varepsilon+L\\,E_{n-1},\\qquad\\text{equivalently}\\\\\nE_n&\\le\\varepsilon\\frac{L^n-1}{L-1}\\ (L\\neq1),\\qquad E_n\\le n\\varepsilon\\ (L=1).\n\\end{aligned}\n$$')
lit('eqbreak__big_BA__mathrm_int_E__theta',
    "$$\\big\\|BA_{\\mathrm{int}}E_\\theta(z)-z'\\big\\|_{L^2}\\;\\le\\;\\|BA_{\\mathrm{int}}\\|\\,\\|e\\|_{L^2}+\\|B(A_{\\mathrm{int}}-A)\\|\\,\\|c\\|_{L^2}+\\delta'.$$",
    "$$\n\\begin{aligned}\n\\big\\|BA_{\\mathrm{int}}E_\\theta(z)-z'\\big\\|_{L^2}\\le{}&\\|BA_{\\mathrm{int}}\\|\\,\\|e\\|_{L^2}\\\\\n&+\\|B(A_{\\mathrm{int}}-A)\\|\\,\\|c\\|_{L^2}+\\delta'.\n\\end{aligned}\n$$")
lit('eqbreak_g_begin_pmatrix_A_C',
    '$$g=\\begin{pmatrix}A&C\\\\ D&B\\end{pmatrix},\\qquad A=P_{\\mathcal R}gP_{\\mathcal R},\\ \\ C=P_{\\mathcal R}gP_{\\mathcal K},\\ \\ D=P_{\\mathcal K}gP_{\\mathcal R},\\ \\ B=P_{\\mathcal K}gP_{\\mathcal K}.$$',
    '$$\n\\begin{aligned}\ng&=\\begin{pmatrix}A&C\\\\ D&B\\end{pmatrix},\\qquad A=P_{\\mathcal R}gP_{\\mathcal R},\\ \\ C=P_{\\mathcal R}gP_{\\mathcal K},\\\\\nD&=P_{\\mathcal K}gP_{\\mathcal R},\\ \\ B=P_{\\mathcal K}gP_{\\mathcal K}.\n\\end{aligned}\n$$')
lit('eqbreak__lVert_W_A_I__mathcal_R_rVer',
    '$$1-T_{RMS}=\\frac{\\lVert WC\\rVert_F}{\\sqrt{\\lVert W(A-I_{\\mathcal R})\\rVert_F^2+\\lVert WC\\rVert_F^2}}=\\frac{\\kappa_W}{\\sqrt{1+\\kappa_W^2}},\\qquad \\kappa_W:=\\frac{\\lVert WC\\rVert_F}{\\lVert W(A-I_{\\mathcal R})\\rVert_F},$$',
    '$$\n\\begin{aligned}\n1-T_{RMS}&=\\frac{\\lVert WC\\rVert_F}{\\sqrt{\\lVert W(A-I_{\\mathcal R})\\rVert_F^2+\\lVert WC\\rVert_F^2}}\n=\\frac{\\kappa_W}{\\sqrt{1+\\kappa_W^2}},\\\\\n\\kappa_W&:=\\frac{\\lVert WC\\rVert_F}{\\lVert W(A-I_{\\mathcal R})\\rVert_F},\n\\end{aligned}\n$$')
lit('eqbreak__text_after_projection_on_th',
    '$$\\mathbb E\\big[(W(A-I)x_R)\\big]^{\\top}\\big(WCx_K\\big)\\;=\\;0\\ \\text{ after projection on the predictable directions},$$',
    '$$\n\\begin{aligned}\n&\\mathbb E\\big[(W(A-I)x_R)\\big]^{\\top}\\big(WCx_K\\big)=0\\\\\n&\\qquad\\text{after projection on the predictable directions},\n\\end{aligned}\n$$')
lit('eqbreak_E_n_le_varepsilon_L_E__n_1_q',
    '$$E_n\\le\\varepsilon+L\\,E_{n-1},\\qquad\\text{so}\\qquad E_n\\le\\varepsilon\\,\\frac{L^n-1}{L-1}\\ (L\\neq1),\\qquad E_n\\le n\\varepsilon\\ (L=1).$$',
    '$$\n\\begin{aligned}\nE_n&\\le\\varepsilon+L\\,E_{n-1},\\qquad\\text{so}\\\\\nE_n&\\le\\varepsilon\\,\\frac{L^n-1}{L-1}\\ (L\\neq1),\\qquad E_n\\le n\\varepsilon\\ (L=1).\n\\end{aligned}\n$$')
lit('eqbreak__rho_tau_nz_g__tau_n_z_',
    '$$\\rho(\\tau)^nz-g_{\\tau^n}z=\\big[\\rho(\\tau)\\big(\\rho(\\tau)^{n-1}z\\big)-g_\\tau\\big(\\rho(\\tau)^{n-1}z\\big)\\big]+\\big[g_\\tau\\big(\\rho(\\tau)^{n-1}z\\big)-g_\\tau\\big(g_{\\tau^{n-1}}z\\big)\\big].$$',
    '$$\n\\begin{aligned}\n\\rho(\\tau)^nz-g_{\\tau^n}z\n&=\\big[\\rho(\\tau)\\big(\\rho(\\tau)^{n-1}z\\big)-g_\\tau\\big(\\rho(\\tau)^{n-1}z\\big)\\big]\\\\\n&\\quad+\\big[g_\\tau\\big(\\rho(\\tau)^{n-1}z\\big)-g_\\tau\\big(g_{\\tau^{n-1}}z\\big)\\big].\n\\end{aligned}\n$$')
lit('eqbreak__big_BA__mathrm_int_E__theta',
    "$$\\big\\|BA_{\\mathrm{int}}E_\\theta(z)-z'\\big\\|_{L^2}\\le\\|BA_{\\mathrm{int}}\\|\\,\\|e\\|_{L^2}+\\|B(A_{\\mathrm{int}}-A)\\|\\,\\|c\\|_{L^2}+\\delta'.$$",
    "$$\n\\begin{aligned}\n\\big\\|BA_{\\mathrm{int}}E_\\theta(z)-z'\\big\\|_{L^2}\\le{}&\\|BA_{\\mathrm{int}}\\|\\,\\|e\\|_{L^2}\\\\\n&+\\|B(A_{\\mathrm{int}}-A)\\|\\,\\|c\\|_{L^2}+\\delta'.\n\\end{aligned}\n$$")

lit_all('fig11_text',
        'Figure 7 shows the interface as built: the readout error over the $(s,v)$ domain, the code magnitude that bounds it, the real-region comparison against the pre-registered threshold, and the cost of an edit.',
        'Figure 7 shows the interface as built: the readout error over the $(s,v)$ domain, the code magnitude that bounds it, the real-region comparison against the pre-registered threshold, and the cost of an edit. **Figure 11** puts the recovery itself on the same footing, as four separate tests: the operator fitted on a shape it never saw against copying on that shape; the hue phase read out of the code against the hue that was rendered, cell by cell; the same error against how much hue the renderer itself put in the pixel; and the attribute the code reports after a scaling against the attribute the scaling realised.')
lit_all('fig12_text',
        'a shared-rate predictor that beats the copy baseline $14.9\\times$ on that carrier loses to it on the real sites ($0.83$ and $0.75$).',
        'a shared-rate predictor that beats the copy baseline $14.9\\times$ on that carrier loses to it on the real sites ($0.83$ and $0.75$). **Figure 12** collects the out-of-sample tests of this section and of §6.6: the closed form used as a predictor across the reachable sites, the sharing term inside and outside the fit split, the rank-resolved benefit of letting channels correct one another, and the shared-rate extrapolation itself.')

lit_all('fig10_text',
        'Table F.10 carries all four arms, three seeds, the parameter counts and the protocol.',
        'Table F.10 carries all four arms, three seeds, the parameter counts and the protocol. Figure 10 gives the read-out error, the two never-fitted increments and the composition defect for each arm.')


def apply(text, kind, start, end, new):
    if kind == 'litall':
        if text.count(start) < 1:
            raise SystemExit(f'operation {start[:60]!r}: literal not found')
        return text.replace(start, new)
    if kind == 'lit':
        if text.count(start) != 1:
            raise SystemExit(f'operation {start[:60]!r}: literal occurs {text.count(start)} times, expected 1')
        return text.replace(start, new, 1)
    if text.count(start) < 1:
        raise SystemExit(f'operation {start[:60]!r}: start marker not found')
    if text.count(start) > 1:
        raise SystemExit(f'operation {start[:60]!r}: start marker not unique ({text.count(start)})')
    i = text.index(start)
    j = text.find(end, i)
    if j < 0:
        raise SystemExit(f'operation {start[:60]!r}: end marker {end[:60]!r} not found after start')
    j += len(end)
    return text[:i] + new + text[j:]


# ============================================================================
#  W2, second half (2026-09-16): the last internal names, and the two promises the appendices could no
#  longer keep once the `*Source:` lines left the paper (D6).
#
#  The operator families were abbreviated O1/O2/O5/O6/O8, the grids `dense5`, the arms z2/ce8, and the
#  sites y3/l2.  Every one of them is replaced by a name a reader can follow; the names are introduced
#  once in Appendix C.5 and used everywhere after, so no table needs a code book.
# ============================================================================
PROMISE_FIXES = [
    # the significance levels of the crossing correlation are in D.7, and the paper no longer carries
    # source files at all (D6), so the pointer is corrected and the second half of the promise dropped
    ('significance levels and source files in Appendix C.7;',
     'significance levels in Appendix D.7;'),
    # the appendix guide promised a source line under every table
    ('Every table carries a source line naming the result file and, where they exist, the scripts that produced it.',
     'Every table states the configuration it was measured on, and every claim carries the scope in which it holds.'),
    # Appendix F promised a column naming the file each number comes from
    ('what was held out, the primary capability statistic with its baseline, and the file the numbers come from.',
     'what was held out, and the primary capability statistic with its baseline.'),
    ("Values are those of the main text's Tables 14 and 15 unless a source line says otherwise; "
     "where a stored file disagrees with a main-text cell, the raw file is preferred and the cell is "
     "marked.",
     "Values are those of the main text's Tables 14 and 15 unless the row states otherwise."),
]

NAME_FIXES = [
    # phrases that carry their own gloss, handled before the bare codes so the gloss is not doubled
    ('| O8 mean gain | O6 random win |', '| conv-res mean gain | random win |'),
    ('O8 = ridge $1\\times1$ plus a $3\\times3$ residual convolution',
     'conv-res is a ridge $1\\times1$ map plus a $3\\times3$ residual convolution'),
    ('O6 = random orthogonal', 'random = orthogonal'),
    ('O2 = unconstrained', 'ridge = unconstrained'),
    ('O1 = global orthogonal', 'Procrustes = global orthogonal'),
    ("sharing O8's architecture", "sharing the conv-res architecture"),
    ("O6 a random orthogonal matrix from the same seed", "random is a random orthogonal matrix from the same seed"),
    ('O5 = content-conditioned (ridge plus a residual MLP)', 'ridge+MLP is content-conditioned (ridge plus a residual MLP)'),
    ('O1, O2, O8, O6 |', 'Procrustes, ridge, conv-res, random |'),
    ('$T_F$ O1/O2/O8/O6', '$T_F$ Procrustes/ridge/conv-res/random', 4),
    ('O8 budget | ridge', 'conv-res budget | ridge'),
]

# (regex, replacement) applied with word boundaries so that `YOLO11n` and `DINOv2` are untouched
CODE_MAP = [
    (r'O1', 'Procrustes'), (r'O2', 'ridge'), (r'O5', 'ridge+MLP'),
    (r'O6', 'random'), (r'O8', 'conv-res'),
    (r'\$y3\$', 'stage 3'), (r'\$y5\$', 'stage 5'), (r'\$y7\$', 'stage 7'),
    (r'y3', 'stage 3'), (r'y5', 'stage 5'), (r'y7', 'stage 7'),
    (r'l1', 'layer 1'), (r'l2', 'layer 2'), (r'l3', 'layer 3'), (r'l4', 'layer 4'),
    (r'z2hue', 'hue-augmented plain CNN'), (r'z2', 'plain CNN'),
    (r'ce8', 'colour-equivariant CNN'), (r'lcer8', 'colour-equivariant residual CNN'),
    (r'ocode', 'code-auxiliary arm'),
    (r'dense5', 'the dense orbit suite'), (r'dense3d', 'the 3-D lattice'),
]


def add_availability(s):
    """Append the code-availability sentence, if a repository has been configured.

    `meta/repo.json` holds the URL and the archival DOI.  While both are empty nothing is added: the
    paper must not cite a repository that does not exist, and a placeholder link is worse than none.
    """
    path = os.path.join(WORK, 'meta', 'repo.json')
    if not os.path.exists(path):
        print('  no meta/repo.json: the paper carries no code-availability statement')
        return s
    cfg = json.load(open(path))
    url, doi = cfg.get('url', '').strip(), cfg.get('doi', '').strip()
    if not url and not doi:
        print('  meta/repo.json has no url/doi: the paper carries no code-availability statement')
        return s
    where = url or ('https://doi.org/' + doi)
    ref = ''
    if url and doi:
        ref = f' (archived at \\texttt{{https://doi.org/{doi}}})'
    sentence = ('\n\n**Code and data availability.** The scripts that produce every table and figure of '
                f'this paper, together with the stored result files they read, are available at '
                f'\\texttt{{{where}}}{ref}. The manuscript states the configuration behind each '
                'number; the repository carries the code, the seeds and the raw outputs.')
    marker = 'The appendices follow in that order.'
    if s.count(marker) != 1:
        raise SystemExit('availability: the appendix guide marker was not found exactly once')
    s = s.replace(marker, marker + sentence, 1)
    print(f'  code-availability statement added ({where})')
    return s


def deinternalise(s):
    """Replace the last internal names by reader-facing ones, and repair two stale promises."""
    for old, new in PROMISE_FIXES:
        if s.count(old) != 1:
            raise SystemExit(f'promise fix {old[:50]!r}: {s.count(old)} occurrences')
        s = s.replace(old, new, 1)
    for item in NAME_FIXES:
        old, new = item[0], item[1]
        want = item[2] if len(item) > 2 else 1
        if s.count(old) != want:
            raise SystemExit(f'name fix {old[:50]!r}: {s.count(old)} occurrences, expected {want}')
        s = s.replace(old, new)
    total = 0
    for pat, name in CODE_MAP:
        rx = re.compile(r'(?<![A-Za-z0-9])' + pat + r'(?![A-Za-z0-9])')
        s, n = rx.subn(name, s)
        total += n
    print(f'  replaced {total} internal name(s)')
    # the parenthetical glosses the tables already carried become redundant once the code is the name
    POST_FIXES = [
        ('plain CNN (plain CNN)', 'plain CNN', 2),
        ('CEConv (colour-equivariant CNN)', 'CEConv', 1),
        ('LCER (colour-equivariant residual CNN)', 'LCER', 1),
        ('code-auxiliary (code-auxiliary arm)', 'code-auxiliary arm', 1),
        ('$+$ hue augmentation (hue-augmented plain CNN)', '$+$ hue augmentation', 1),
        ('$stage 3,stage 5,stage 7$', 'stage 3, stage 5, stage 7', 1),
    ]
    for old, new, want in POST_FIXES:
        if s.count(old) != want:
            raise SystemExit(f'post fix {old[:40]!r}: {s.count(old)} occurrences, expected {want}')
        s = s.replace(old, new)
    # the last laboratory traces: result-file keys quoted in the appendices, the script column of the
    # rendering-grid table, and column names set in code font where italics are the convention
    TRACE_FIXES = [
        ('Table 3, `transport_theory.json` | linear sites |', 'Table 3 | linear sites |', 1),
        ('(`realised 0.174`, `error 0.0143`)',
         '(realised report change $0.174$, read-out error $0.0143$)', 1),
        ('ResNet-18 `layer 2`', 'ResNet-18 layer 2', 1),
        ('60 epochs, `imgsz` 224, batch 16, `mosaic/mixup/hsv_h/hsv_s/hsv_v/fliplr/flipud` all $0$, seed $0$',
         '60 epochs at $224$ px, batch 16, with every augmentation (mosaic, mixup, hue, saturation, value, '
         'horizontal and vertical flip) disabled, seed $0$', 1),
        ('`n_rows_used_for_fit` $=9800$', 'fitted rows $=9800$', 1),
        ('(`split_overlap` $0$)', '(no fit/held-out overlap)', 1),
        ('(the harness records `intervention_reaches_consumer` and `random_operator_logit_change`)',
         '(the harness records whether the intervention reaches the consumer and whether a random '
         'operator changes the logits)', 1),
        ('`direct` is fitted at the composed parameter (upper reference); `single` is the $t_1$ operator evaluated on the composite.',
         '*direct* is fitted at the composed parameter (upper reference); *single* is the $t_1$ operator '
         'evaluated on the composite.', 1),
        ('`transfer` is the fraction of the no-op', '*transfer* is the fraction of the no-op', 1),
        ('`readout` is the ridge probe fitted on four shapes',
         '*readout* is the ridge probe fitted on four shapes', 1),
        ('CPU-only, four threads (`OMP_NUM_THREADS=4`, `nice -n 15`), GPU deliberately untouched',
         'CPU-only, four threads at low priority, GPU deliberately untouched', 1),
        ('$d^2/K=428.0$ at `resnet50|heat|depth4`', '$d^2/K=428.0$ at ResNet-50, heat, depth 4', 1),
        ('| grid | shapes | hues | $(s,v)$ | variants/cell | script |\n|---|---|---|---|---|---|\n',
         '| grid | shapes | hues | $(s,v)$ | variants/cell |\n|---|---|---|---|---|\n', 1),
    ]
    for old, new, want in TRACE_FIXES:
        if s.count(old) != want:
            raise SystemExit(f'trace fix {old[:50]!r}: {s.count(old)} occurrences, expected {want}')
        s = s.replace(old, new)
    for old, new, want in [
            ('`resnet18_layer 2`', 'ResNet-18 layer 2', 1),
            ('`yolo11n_stage 3`', 'YOLO11n stage 3', 1),
            ('launched as `OMP_NUM_THREADS=4 nice -n 15`', 'launched on four threads at low priority', 1),
            ('`phase cosine`', '*phase cosine*', 1),
            ('`plain CNN`', 'plain CNN', 1),
            ('`not recorded`', '*not recorded*', 1),
    ]:
        if s.count(old) != want:
            raise SystemExit(f'last trace {old!r}: {s.count(old)} occurrences, expected {want}')
        s = s.replace(old, new)
    # the script column values leave with the column
    for pat in (r' \| `01` \|', r' \| `30` \|', r' \| `41` \|', r' \| `115` \|'):
        s = re.sub(pat, ' |', s)
    for code in ('`01`', '`30`', '`41`', '`115`'):
        if code in s:
            raise SystemExit(f'the script column value {code} survived')
    left = sorted({m.group(0) for m in re.finditer(
        r'(?<![A-Za-z0-9])(?:O[1-9]|y[0-9]|l[1-4]|z2|ce8|lcer8|ocode|dense5|dense3d)(?![A-Za-z0-9])', s)})
    if left:
        raise SystemExit(f'internal name(s) still in the manuscript: {left}')
    return s


# ============================================================================
#  Final stage (W5): citations and float numbering.
#
#  Two normalisations that a submission needs and the manuscript did not carry:
#
#  (1) CITATIONS.  The manuscript cites by author-year-venue ("(Gruver et al., ICLR 2023)") while the
#      reference list is a numbered list, so a reader of the PDF cannot resolve a citation.  The table
#      below rewrites every citation site to an IEEE-style numeric marker.  The numbers are NOT written
#      here: each site names its bibliography key, the keys are numbered by first appearance in the
#      assembled text (IEEE order), and the reference list is emitted in that order.  A site whose
#      pattern is not found, or found a different number of times, fails the build.
#
#  (2) FLOAT NUMBERS.  Tables and figures were numbered in the order they were written, not in the
#      order the text cites them (Table 6 is the first table cited, Table 2 the third).  Both are
#      renumbered by first appearance here; the plural forms are handled explicitly because a range
#      such as "Tables 10--11" is no longer contiguous after renumbering.
# ============================================================================
import collections

PLACEHOLDER = '[[cite:%s]]'

# (regex, replacement template, keys, expected number of matches)
CITE_SITES = [
    # -- §2.1 transformations in existing representations
    (r'Lenc & Vedaldi \(CVPR 2015\)', 'Lenc and Vedaldi {c}', ['lenc2015'], 2),
    (r'\(Gruver et al\., ICLR 2023\)', '{c}', ['gruver2023'], 1),
    (r'Bruintjes et al\. \(CVPRW 2023\)', 'Bruintjes et al. {c}', ['bruintjes2023'], 2),
    (r'Romero & Lohit \(NeurIPS 2022\)', 'Romero and Lohit {c}', ['romero2022'], 2),
    (r'Brehmer et al\. \(TMLR 2024\)', 'Brehmer et al. {c}', ['brehmer2024'], 1),
    (r'He et al\. \(2026\)', 'He et al. {c}', ['he2026'], 1),
    (r'\(Dangovski et al\., ICLR 2022; Wang et al\., NeurIPS 2024\)', '{c}',
     ['dangovski2022', 'wang2024'], 1),
    (r'Gruver et al\. \(ICLR 2023\)', 'Gruver et al. {c}', ['gruver2023'], 1),
    # -- §2.2 symmetry-based and equivariant representations
    (r'\(Cohen & Welling, ICML 2016\)', '{c}', ['cohen2016'], 1),
    (r'\(Weiler & Cesa, NeurIPS 2019\)', '{c}', ['weiler2019'], 1),
    (r'\(Lengyel et al\., NeurIPS 2023\)', '{c}', ['lengyel2023'], 1),
    (r'\(Yang et al\., ICLR 2025\)', '{c}', ['yang2025'], 1),
    (r'\(Yang et al\., ICML 2026\)', '{c}', ['yang2026'], 1),
    (r'\(Pacini et al\., ICLR 2024\)', '{c}', ['pacini2024'], 1),
    (r'\(Higgins et al\., 2018\)', '{c}', ['higgins2018'], 1),
    (r'Cohen & Welling \(ICML 2016\)', 'Cohen and Welling {c}', ['cohen2016'], 1),
    (r'Weiler & Cesa \(NeurIPS 2019\)', 'Weiler and Cesa {c}', ['weiler2019'], 1),
    (r'Lengyel et al\. \(NeurIPS 2023\)', 'Lengyel et al. {c}', ['lengyel2023'], 1),
    (r'Yang et al\. \(ICLR 2025\)', 'Yang et al. {c}', ['yang2025'], 1),
    (r'Yang et al\. \(ICML 2026\)', 'Yang et al. {c}', ['yang2026'], 1),
    (r'Higgins et al\. \(2018\)', 'Higgins et al. {c}', ['higgins2018'], 1),
    # -- §2.3 colour and perceptual transformations
    (r'\(Koenderink, 1984; Duits et al\., 2004\)', '{c}', ['koenderink1984', 'duits2004'], 1),
    (r'Koenderink \(1984\)', 'Koenderink {c}', ['koenderink1984'], 1),
    (r'Duits et al\. \(2004\)', 'Duits et al. {c}', ['duits2004'], 1),
    # -- §2.4 causal abstraction and latent linearisation
    (r'\(Geiger et al\., ICML 2022; and its theoretical foundation, JMLR 26\(83\), 2025\)', '{c}',
     ['geiger2022', 'geiger2025'], 1),
    (r'Geiger et al\. \(ICML 2022; JMLR 2025\)', 'Geiger et al. {c}', ['geiger2022', 'geiger2025'], 1),
    (r'\(Park et al\., NeurIPS 2024\)', '{c}', ['park2024'], 1),
    (r'Park et al\. \(NeurIPS 2024\)', 'Park et al. {c}', ['park2024'], 1),
    (r'\(Nadaf, 2026\)', '{c}', ['nadaf2026'], 1),
    (r'Nadaf \(2026\)', 'Nadaf {c}', ['nadaf2026'], 1),
    (r'\(Venkatesh & Kurapath, 2026\)', '{c}', ['venkatesh2026'], 1),
    (r'Venkatesh & Kurapath \(2026\)', 'Venkatesh and Kurapath {c}', ['venkatesh2026'], 1),
    (r'Wehner et al\. \(2025\)', 'Wehner et al. {c}', ['wehner2025'], 1),
    (r'\(Asiaee, UAI 2026\)', '{c}', ['asiaee2026'], 1),
    (r'\(Karnam & Sathish, 2026\)', '{c}', ['karnam2026'], 1),
    (r'\(Buchholz, 1994\)', '{c}', ['buchholz1994'], 1),
    (r'\(Mac Lane & Birkhoff, 1999; Milner, 1989\)', '{c}', ['maclane1999', 'milner1989'], 1),
    # -- protocol section
    (r'\(Oquab et al\., TMLR 2024\)', '{c}', ['oquab2024'], 1),
]

# The bibliography, IEEE style, keyed.  Contents are those of the reference list of paper_v15.md with
# author order, punctuation and venue formatting normalised; `lusch2018` and `serre1977` are new and
# are the two citations the Koopman and representation-theory sentences of §2.4 and §2.2 were missing.
BIB = {
 'asiaee2026': 'A. Asiaee, "Certified Interventional Fidelity: Anytime-Valid, Adaptive Evaluation of Causal Claims in Mechanistic Interpretability," in Proc. UAI, 2026. arXiv:2607.08349.',
 'brehmer2024': 'J. Brehmer, S. Behrends, P. de Haan, and T. Cohen, "Does equivariance matter at scale?" TMLR, 2024. arXiv:2410.23179.',
 'bruintjes2023': 'R.-J. Bruintjes, T. Motyka, and J. van Gemert, "What Affects Learned Equivariance in Deep Image Recognition Models?" in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 4839--4847, 2023. arXiv:2304.02628.',
 'buchholz1994': 'P. Buchholz, "Exact and ordinary lumpability in finite Markov chains," Journal of Applied Probability, vol. 31, no. 1, pp. 59--75, 1994.',
 'cohen2016': 'T. Cohen and M. Welling, "Group Equivariant Convolutional Networks," in Proc. ICML, 2016.',
 'dangovski2022': 'R. Dangovski, L. Jing, C. Loh, S. Han, A. Srivastava, et al., "Equivariant Contrastive Learning," in Proc. ICLR, 2022. arXiv:2111.00899.',
 'duits2004': 'R. Duits, L. Florack, J. de Graaf, and B. ter Haar Romeny, "On the axioms of scale space theory," Journal of Mathematical Imaging and Vision, vol. 20, no. 3, pp. 267--298, 2004.',
 'geiger2025': 'A. Geiger, D. Ibeling, A. Zur, M. Chaudhary, S. Chauhan, J. Huang, A. Arora, Z. Wu, N. Goodman, C. Potts, and T. Icard, "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability," Journal of Machine Learning Research, vol. 26, no. 83, pp. 1--64, 2025. arXiv:2301.04709.',
 'geiger2022': 'A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. Goodman, and C. Potts, "Inducing Causal Structure for Interpretable Neural Networks," in Proc. ICML, PMLR 162:7324--7338, 2022.',
 'gruver2023': 'N. Gruver, M. Finzi, M. Goldblum, and A. G. Wilson, "The Lie Derivative for Measuring Learned Equivariance," in Proc. ICLR, 2023. arXiv:2210.02984.',
 'he2026': 'T. He, A. H. Williams, and S. E. Harvey, "How Data Augmentation Shapes Neural Representations," 2026. arXiv:2605.15306.',
 'higgins2018': 'I. Higgins, D. Amos, D. Pfau, S. Racaniere, L. Matthey, D. Rezende, and A. Lerchner, "Towards a Definition of Disentangled Representations," arXiv preprint arXiv:1812.02230, 2018.',
 'karnam2026': 'A. G. Karnam and T. Sathish, "EditCLEVR: A Paired-Scene Intervention Benchmark for Compositional Faithfulness of Object-Centric Representations," 2026. arXiv:2607.22705.',
 'koenderink1984': 'J. J. Koenderink, "The structure of images," Biological Cybernetics, vol. 50, no. 5, pp. 363--370, 1984.',
 'lenc2015': 'K. Lenc and A. Vedaldi, "Understanding Image Representations by Measuring Their Equivariance and Equivalence," in Proc. CVPR, 2015.',
 'lengyel2023': 'A. Lengyel, O. Strafforello, R.-J. Bruintjes, A. Gielisse, and J. van Gemert, "Color Equivariant Convolutional Networks," in Proc. NeurIPS, 2023. arXiv:2310.19368.',
 'lusch2018': 'B. Lusch, J. N. Kutz, and S. L. Brunton, "Deep learning for universal linear embeddings of nonlinear dynamics," Nature Communications, vol. 9, art. 4950, 2018.',
 'maclane1999': 'S. Mac Lane and G. Birkhoff, Algebra, 3rd ed. Providence, RI: AMS Chelsea Publishing, 1999.',
 'milner1989': 'R. Milner, Communication and Concurrency. Englewood Cliffs, NJ: Prentice Hall, 1989.',
 'nadaf2026': 'M. S. B. Nadaf, "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens," 2026. arXiv:2604.02608.',
 'oquab2024': 'M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR, 2024. arXiv:2304.07193.',
 'pacini2024': 'M. Pacini, X. Dong, B. Lepri, and G. Santin, "A Characterization Theorem for Equivariant Networks with Point-wise Activations," in Proc. ICLR, 2024. arXiv:2401.09235.',
 'park2024': 'K. Park, Y. J. Choe, and V. Veitch, "The Linear Representation Hypothesis and the Geometry of Large Language Models," in Proc. NeurIPS, 2024. arXiv:2311.03658.',
 'romero2022': 'D. W. Romero and S. Lohit, "Learning Partial Equivariances from Data," in Proc. NeurIPS, 2022. arXiv:2111.10571.',
 'serre1977': 'J.-P. Serre, Linear Representations of Finite Groups, Graduate Texts in Mathematics 42. New York, NY: Springer, 1977.',
 'venkatesh2026': 'S. Venkatesh and A. M. Kurapath, "On the Non-Identifiability of Steering Vectors in Large Language Models," 2026. arXiv:2602.06801.',
 'wang2024': 'Y. Wang, K. Hu, S. Gupta, Z. Ye, Y. Wang, and S. Jegelka, "Understanding the Role of Equivariance in Self-supervised Learning," in Advances in Neural Information Processing Systems 37 (NeurIPS), 2024. arXiv:2411.06508.',
 'wehner2025': 'J. Wehner, S. Abdelnabi, D. Tan, D. Krueger, and M. Fritz, "Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models," 2025. arXiv:2502.19649.',
 'weiler2019': 'M. Weiler and G. Cesa, "General E(2)-Equivariant Steerable CNNs (e2cnn)," in Proc. NeurIPS, 2019. arXiv:1911.08251.',
 'yang2025': "Y. Yang, F. O'Mahony, and C. Allen-Blanchette, \"Learning Color Equivariant Representations,\" in Proc. ICLR, 2025. arXiv:2406.09588.",
 'yang2026': 'Y. Yang, Z. Xu, Y. Li, and C. Allen-Blanchette, "A Hypertoroidal Covering for Perfect Color Equivariance (T$^3$CEN)," in Proc. ICML, 2026. arXiv:2603.04256.',
}



def apply_citations(s):
    """Rewrite every citation site to a keyed placeholder, number the keys, emit the reference list."""
    # The old reference list is replaced, not edited, so it is cut out of every scan below.  Head and
    # tail are rejoined through a sentinel rather than by offset: the replacements change the length of
    # the text, so a remembered offset would slice the appendix mid-sentence.
    i_ref, i_app = s.index('\n## References'), s.index('\n## Appendices')
    split = '\x02appendix-split\x02'
    if split in s:
        raise SystemExit('sentinel collision')
    s = s[:i_ref] + split + s[i_app:]
    key_order = []
    for pat, tpl, keys, expect in CITE_SITES:
        rx = re.compile(pat)
        found = len(rx.findall(s))
        if found != expect:
            raise SystemExit(f'citation site {pat!r}: {found} matches, expected {expect}')
        for k in keys:
            if k not in BIB:
                raise SystemExit(f'citation key {k} has no bibliography entry')
        marker = '{c}'
        def repl(m, keys=keys, tpl=tpl):
            ph = PLACEHOLDER % '|'.join(keys)
            return tpl.replace(marker, ph)
        s = rx.sub(repl, s)
    # number the keys by first appearance
    for m in re.finditer(r'\[\[cite:([a-z0-9|]+)\]\]', s):
        for k in m.group(1).split('|'):
            if k not in key_order:
                key_order.append(k)
    num = {k: i + 1 for i, k in enumerate(key_order)}
    missing = [k for k in BIB if k not in num]
    if missing:
        raise SystemExit(f'reference(s) never cited: {missing}')

    def num_repl(m):
        return ', '.join('[%d]' % num[k] for k in m.group(1).split('|'))
    s = re.sub(r'\[\[cite:([a-z0-9|]+)\]\]', num_repl, s)

    # leftover author-year sites are a failure, not a warning
    leftovers = []
    for pat in (r'\([A-Z][^()]{0,70}?(?:19|20)\d\d[^()]{0,40}?\)',
                r'[A-Z][a-z]+ (?:&|and) [A-Z][a-z]+ \([^()]{0,40}\)',
                r'[A-Z][a-z]+ et al\. \([^()]{0,40}\)'):
        for m in re.finditer(pat, s):
            txt = m.group(0)
            if re.search(r'(19|20)\d\d', txt) or re.search(
                    r'ICLR|ICML|NeurIPS|CVPR|TMLR|JMLR|UAI|ICCV|ECCV|arXiv', txt):
                leftovers.append(txt)
    if leftovers:
        raise SystemExit(f'unconverted citation(s): {leftovers[:6]}')

    # rebuild the reference list in citation order, between the two halves
    head, tail = s.split(split)
    refs = '\n## References\n\n' + '\n'.join(
        f'- [{num[k]}] {BIB[k]}' for k in key_order) + '\n'
    return head + refs + tail, num


def strip_sources(s, out_path):
    """Move every `*Source:` line out of the paper and into the internal traceability ledger.

    A submission must not print the repository paths of the scripts and result files behind a table
    (D6): the numbers stay in the paper, the provenance moves here.  Each line is recorded against the
    caption or section it belongs to, the ledger is written to `meta/SOURCES.md`, and the manuscript is
    rebuilt without the lines.  A `*Source:` line that survives is a build failure.
    """
    lines = s.split('\n')
    kept, ledger = [], []
    where = '(front matter)'
    for ln in lines:
        m = re.match(r'^\*\*Table ([A-Z]?\.?\d+[^.]*)\.\*\*', ln) or \
            re.match(r'^\*\*Figure (\d+)\.\*\*', ln)
        if m:
            where = ('Table ' if ln.startswith('**Table') else 'Figure ') + m.group(1)
        elif ln.startswith('### ') or ln.startswith('## '):
            where = ln.lstrip('#').strip()
        m_src = re.search(r'\*Sources?:', ln)
        if m_src:
            # the note is either a paragraph of its own or the trailing element of the paragraph that
            # carries the table's numbers; either way it is removed and recorded against its caption
            head, note = ln[:m_src.start()], ln[m_src.end():]
            ledger.append((where, note.strip()))
            if not head.strip():
                continue
            ln = head.rstrip()
        kept.append(ln)
    s = '\n'.join(kept)
    s = re.sub(r'\n{3,}', '\n\n', s)
    if re.search(r'\*Sources?:', s):
        raise SystemExit('a *Source: line survived the strip')
    with open(out_path, 'w') as fh:
        fh.write('# Provenance of every number in the manuscript (internal)\n\n')
        fh.write('Generated by `scripts/264_integrate_v17.py` from the `*Source:` lines of the '
                 'manuscript, which are removed from the paper itself (decision D6).  The paper keeps '
                 'the numbers; this file keeps the path from each table or figure back to the result '
                 'file and the script that produced it.\n\n')
        fh.write('| where | source |\n|---|---|\n')
        for w, t in ledger:
            fh.write('| %s | %s |\n' % (w, t.replace('|', '\\|')))
    print(f'  moved {len(ledger)} source line(s) to meta/SOURCES.md')
    return s


def renumber_floats(s, fig_map):
    """Renumber main-text tables and figures by first appearance, then normalise plural references.

    Table numbers are derived from the text: the defined numbers are those of the bold captions, the
    cited numbers are every other mention, and the new order is the order in which each number is first
    cited.  Figure numbers are derived the same way, but the set of defined figures comes from
    `captions.json` (whose `was` field records the pre-renumbering number), because the figure captions
    are inserted after this pass.  The derived figure map must equal the one `captions.json` declares,
    so the plate order, the running text and the script outputs can never drift apart.
    """
    def derive(kind, defined, ref_re):
        first, seen = [], set()
        for m in re.finditer(ref_re, s):
            n = int(m.group(1))
            if n not in seen:
                seen.add(n)
                first.append(n)
        if set(first) != set(defined):
            raise SystemExit(f'{kind}: cited {sorted(set(first))} != defined {sorted(defined)}')
        return {old: i + 1 for i, old in enumerate(first)}

    tdefined = [int(n) for n in re.findall(r'\*\*Table (\d+)\.\*\*', s)]
    if len(set(tdefined)) != len(tdefined):
        raise SystemExit('table: duplicate caption number')
    tmap = derive('table', tdefined, re.compile(r'\bTable (\d+)\b'))
    fmap = derive('figure', set(fig_map), re.compile(r'\bFigure (\d+)\b'))
    if fmap != fig_map:
        raise SystemExit(f'figure numbering disagrees with captions.json: derived {fmap} '
                         f'vs declared {fig_map}; reorder scripts/280_make_caption_file.py')

    def plural(m):
        """A 'Tables ...' phrase: remap the plain numbers, keep lettered appendix labels, re-render."""
        body = m.group(1)
        plain = [int(n) for n in re.findall(r'\d+', body)]
        lettered = re.findall(r'[A-Z]\.\d+', body)
        new = sorted({tmap[p] for p in plain})
        if len(new) >= 3 and new == list(range(new[0], new[-1] + 1)):
            rendered = f'Tables {new[0]}\u2013{new[-1]}'
        elif len(new) == 2:
            rendered = f'Tables {new[0]} and {new[1]}'
        else:
            rendered = 'Tables ' + ', '.join(str(n) for n in new)
        if lettered:
            rendered += ' and ' + ' and '.join(lettered)
        return rendered

    s = re.sub(r'\bTables ((?:\d+|[A-Z]\.\d+)(?:\s*(?:,|and|\u2013|-)\s*(?:\d+|[A-Z]\.\d+))*)', plural, s)
    s, n1 = re.subn(r'\bTable (\d+)\b', lambda m: 'Table %d' % tmap[int(m.group(1))], s)
    s, n2 = re.subn(r'\bFigure (\d+)\b', lambda m: 'Figure %d' % fmap[int(m.group(1))], s)
    print(f'  renumbered {n1} table and {n2} figure references')
    return s


def main():
    s = open(SRC).read()
    n0 = len(s.split())
    applied = []
    for kind, name, a, b, new in OPS:
        try:
            s2 = apply(s, kind, a, b, new)
        except SystemExit as exc:
            print('FAILED', exc)
            sys.exit(1)
        if s2 == s:
            raise SystemExit(f'operation {name}: no change made')
        s = s2
        applied.append(name)
    # W5: citations -> IEEE numbers (with the reference list rebuilt in citation order), then the
    #     main-text table and figure numbers reordered by first appearance.
    s, cnum = apply_citations(s)
    caps_pre = json.load(open(os.path.join(WORK, 'paper', 'figures_final', 'captions.json')))
    s = renumber_floats(s, {c['was']: c['n'] for c in caps_pre})
    s = deinternalise(s)
    s = add_availability(s)
    s = strip_sources(s, os.path.join(WORK, 'meta', 'SOURCES.md'))
    print(f'  numbered {len(cnum)} references by first citation')

    # W1: the figure section is the twelve vector figures, with the captions from captions.json
    caps = json.load(open(os.path.join(WORK, 'paper', 'figures_final', 'captions.json')))
    figblock = '\n## Figures\n\n' + '\n\n'.join(
        f"**Figure {c['n']}.** {c['caption']}" for c in caps) + '\n'
    s = re.sub(r'\n## Figures\n.*?(?=\n## References\n)', lambda m: figblock, s, flags=re.S)

    # structural normalisation: a markdown table must not contain a blank line
    s = re.sub(r'(\n\|[^\n]*)\n\n(\|)', r'\1\n\2', s)
    open(DST, 'w').write(s)
    out = {
        'source': 'paper/archive/paper_v15.md',
        'target': 'paper/paper_v17.md',
        'operations': len(applied),
        'operation_names': applied,
        'words_source': n0,
        'words_target': len(s.split()),
        'references': len(cnum),
        'citation_order': sorted(cnum, key=lambda k: cnum[k]),
    }
    os.makedirs(os.path.join(WORK, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(WORK, 'results', 'v17_integration.json'), 'w'), indent=1)
    print(f"applied {len(applied)} operations; words {n0} -> {out['words_target']}")
    print('-> paper/paper_v17.md')


if __name__ == '__main__':
    main()
