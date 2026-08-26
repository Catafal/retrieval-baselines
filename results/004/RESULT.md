# Experiment 004 — does a better extractor move the graph result

**Answer: yes, a long way, and it removes the finding 003 was built on.**

## The arms

| | HotpotQA R@2 | 2Wiki R@2 |
|---|---|---|
| BM25 | 0.5490 | 0.5164 |
| dense (bge) | 0.6957 | 0.6557 |
| graph, GLM-4.7-Flash (004) | **0.3699** | **0.3770** |
| graph, oracle extractor (003) | 0.3344 | — |
| graph, spaCy (003) | 0.2148 | 0.2734 |

Empty results fell from 17.68% to 9.89% on HotpotQA and from 41.07% to 32.09% on 2Wiki.

## What was registered, and what happened

**The registered ceiling was wrong.** Protocol 004 §2 stated, before running, that 003's oracle
bounded this experiment to roughly [0.2148, 0.3344] because "a real extractor cannot beat a
perfect one". The arm scored 0.3699 and passed it by 0.0355.

The oracle used corpus document titles as entities: a perfect linker over a **fixed entity set**.
It bounded that definition of entity, not extraction. GLM returns about 10 entities per passage
where titles give one, and the denser graph reaches further. The sentence was false and the
experiment falsified it.

**Prediction A is not supported on any of the six cells, and 003's was supported on all six.**

| cell | 003 (spaCy) | 004 (GLM) |
|---|---|---|
| ndcg@10 exact | +0.1216 supported | −0.0110 not supported |
| ndcg@10 stripped | +0.0903 supported | −0.0266 not supported |
| recall@2 exact | +0.0985 supported | +0.0033 not supported |
| recall@2 stripped | +0.0716 supported | −0.0102 not supported |
| recall@5 exact | +0.1568 supported | +0.0035 not supported |
| recall@5 stripped | +0.1182 supported | −0.0159 not supported |

This is registered failure mode 1, in the words the protocol used: *the advantage flattens rather
than rises, meaning 003's differential was an extraction artefact and the 003 finding is
weakened.* Say so, and it is said in `results/003/corrections.md` C11.

**Prediction B holds.** The negative control still reports `confirmed_no_advantage` on all six
cells for both arms, so the machinery that would have caught a graph winning everywhere is
working.

**Prediction C is refuted again, harder.** On 2Wiki the arm still does its best work where no
traversal is needed: −0.3416 R@2 against 003's −0.2587, same direction, Holm p 0.0024.

**Prediction D is refuted.** Corrected shrink is 3.97 points against a registered threshold of 10.

## The finding

Extraction quality was the binding constraint on this arm's overall performance and had nothing
to do with its class differential. A 72% gain in R@2 and the complete disappearance of the
between-class advantage arrived in the same run, from the same change.

003 read the differential as evidence that a graph reaches documents a query does not name. With
a better extractor the arm reaches far more documents and the differential is gone. The honest
reading is that the differential measured how badly spaCy extracted the queries whose gold
documents were named, not how well a graph traverses.

BM25 still wins both corpora by a wide margin, and the oracle result means that gap is not a
ceiling either.

## Cost

110,068 passages and 17,230 queries extracted, ~$10 total, about 5 hours wall clock. The
`extraction-usage.json` artifact records $3.80 because two earlier runs crashed before writing it;
the total is derived from the measured $0.0000866 per passage.
