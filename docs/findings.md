# Findings

A place to record what you learn while post-training, evaluating, or serving GR00T N2
with this repository, in a form that someone else — or an agent, or you in three months
— can act on without re-deriving it. The `autoresearch` and `autodebug` skills read this
file before they start and append to it when they finish.

This file ships nearly empty on purpose. The two entries under
[Recorded findings](#recorded-findings) are worked examples of the format; the knowledge
that matters here is what you learn on your own data and hardware, which nobody upstream
can write for you. An earlier archive of informal notes from previous GR00T generations
was reviewed against the test below and none of it passed; it remains in git history at
`oss/docs/research-notes.md` in commit `c62c2760fe1` for anyone who wants it.

## What belongs here

A finding earns its place by passing four questions. Each is shown with something that
passed and something that was cut while this file was being seeded.

**1. Would a reader hit the problem before they knew to look at the code?** A finding
is not a summary of a source file; it is the thing you wish you had been told. The
closed-loop runner passing `num_envs=1` is one grep away, and the policy adapter raises
a message saying so when you change it — that was cut. A missing camera view being
silently replaced by a zero tensor deep inside the processor, so that a wrong rig scores
badly without any error — that stayed.

**2. Can the reader reach the scenario?** Only record situations reached through this
repository's own commands, datasets you can name, and hardware the reader might have.
Findings about NVIDIA's internal CI gates, internal benchmark variants, or internal
robot rigs were cut even when true, because a reader cannot enter the scenario or check
the number.

**3. Is the evidence something the reader can open?** Cite a file and line in this
repository, a config the reader can read, or a run the reader can reproduce. A finding
whose evidence is an internal document is a rumor to everyone outside.

**4. Is it a fact, or a choice dressed as advice?** "We ran 20 episodes per task because
that is what the CI budget allowed" is a fact about a budget, not guidance about episode
counts. Record what was chosen and what it does and does not show; do not tell the
reader what to do unless a measurement backs it. When you find yourself writing "is
enough" or "you should", check whether you measured the alternative.

Two consequences follow. A finding that restates arithmetic or a well-documented
behaviour of a dependency does not qualify. And a finding that a code change makes
obsolete should be deleted, not amended: this file has no mechanism to track the code.

## How to write one

Every entry has the same shape. Fill every field; a claim with no `Use` is data dumping,
and a claim with no `Limits` is an overclaim.

- **Heading** — the claim in one line, specific enough to be wrong.
- **Kind** — one of four. *measured-result*: a number produced by a run you can name.
  *constraint*: a rule the code imposes. *design-decision*: a value or approach that
  was chosen, recorded without implying the alternative was measured. *caveat*: a way
  of reading something wrong. Do not inflate; most of what looks like knowledge in a
  config tree is a design-decision.
- **Confidence** with a reason — high, medium, or low, followed by one sentence on what
  makes it that and not higher.
- **As of** — the date and, when it matters, the commit or release you checked against.
- **Claim** — two to four sentences. State the thing and where it lives.
- **Use** — how a reader changes what they do because of this. Written as what the
  finding shows, not as instruction, unless a measurement backs the instruction.
- **Limits** — where it stops holding. Part of the claim, not a footnote.
- **Evidence** — file and line, config key, or reproducible command. Keep it to things
  the reader can open.

Three writing rules that were each learned by getting them wrong:

- Definitions say what a thing is, not how it behaves. Behaviour is a claim and needs
  its own `Limits` and evidence.
- Introduce every term the first time it appears, or the reader will attach it to
  whatever similar name is nearest. If the same word names two knobs — `action_horizon`
  in a training config is the predicted chunk length, in the open-loop evaluator it is
  how much of that chunk is read back — say so where the collision happens.
- Quote whole lines. Trimming a comment to the clause you need is how a quote ends up
  saying something its source does not.

### Template

```markdown
### <claim in one line>

*<kind>* · *<confidence> confidence* · as of <YYYY-MM-DD>

<Claim: two to four sentences.>

**Use.** <What a reader does differently.>

**Limits.** <Where it stops holding.>

**Evidence.** <file:line, config key, or command.>

**Why this confidence.** <One sentence.>
```

## Recorded findings

Seeded with two worked examples. Add yours below them; keep the newest at the bottom so
the file reads in the order things were learned.

### A missing camera view is zero-filled, not rejected

*constraint* · *high confidence* · as of 2026-09-23

The N2 processor compares the camera views it receives against the model's configured
maximum (`max_n_cameras`, 4 in the public pretrain config). When fewer arrive, it
appends zero tensors to make up the count and marks them in a padding mask. Nothing
raises. A rollout rig that supplies three of a checkpoint's four training views
therefore runs to completion and scores badly, with no error pointing at the camera.

**Use.** When a post-trained checkpoint scores far below what its training loss
suggests, diff the video keys the rollout supplies against the checkpoint's training
modality config before touching anything else. This matters most for a client of the
policy server, which assembles observations itself and gets no schema check from the
server.

**Limits.** The padding path exists because some embodiments legitimately train with
fewer views than the maximum, and the mask tells the model which slots are padding; a
view that is absent at inference but was present at training is a distribution shift
the mask does not repair. No measured degradation figure for a missing view is recorded
here.

**Evidence.** `groot/core/training/models/n2/processor.py:385-387` (the zero-fill),
`groot/infra/experiments/training/configs/n2_oss_pretrain.py` (`max_n_cameras`).

**Why this confidence.** The zero-fill is three unconditional lines; the consequence is
inferred from them, not from a reported incident.

### An embodiment tag names a data contract, not a robot

*constraint* · *high confidence* · as of 2026-09-23

`--embodiment-tag` is required by every training and evaluation command here, and the
same physical platform carries several tags. The XDOF family, which the YAM examples
use, has five that all combine relative end-effector actions with relative joint actions
and differ only in what rides alongside: whether per-frame subtask language annotations
are included, whether the task language is the original instruction or a longer caption
regenerated by a vision-language model, whether end-effector keys follow the UMI
dataset's naming convention, and whether a fourth bottom camera is present. RoboCasa has
eleven.

**Use.** Pick the tag by matching your dataset's actual keys and annotation fields
against the tag's docstring, not by recognizing the robot. A neighbouring tag loads
without complaint and then reads the wrong fields.

**Limits.** The tag definitions state what each variant changes but not what happens
when you pick the wrong one; the silent-mismatch consequence is inferred from the
schema's shape, not from a reported incident.

**Evidence.** `groot/core/data/schema/embodiment_tags.py`, the `XDOF_*` and
`ROBOCASA_*` members and their docstrings.

**Why this confidence.** The variants and the reasons they exist are documented in the
schema itself, including one that exists only to rename keys.
