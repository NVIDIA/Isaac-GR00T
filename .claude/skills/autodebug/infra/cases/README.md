# Verified AutoDebug cases

These skill-local records are the runtime references used by AutoDebug.

Only independently reproduced cases with a failing-before and passing-after check belong
in this directory.

The initial OSS release keeps the runtime library empty while public cases are validated.

Add one JSON file per verified case:

```json
{
  "version": 1,
  "id": "origin-short-name",
  "status": "verified",
  "title": "short symptom",
  "source": "durable public issue, commit, or artifact locator",
  "scope": ["affected version(s), robot, benchmark, or environment"],
  "tags": ["data", "preprocessing"],
  "match_terms": ["distinctive symptom or error text"],
  "symptom": "what failed",
  "first_test": "the smallest test that separated the leading causes",
  "cause": "the confirmed cause and applicability boundary",
  "fix": "the exact verified change",
  "compute": "hardware, dependencies, runtime, and required assets",
  "verification": "before failure, after pass, and related regression check"
}
```

Required evidence:

- immutable source locator and affected GR00T version;
- failing command, fixture, or controlled incident artifact;
- exact one-factor intervention;
- the same acceptance check before and after;
- at least one regression check;
- compute and artifact locations;
- review confirming `status: verified`.

Keep partial, unresolved, private, and advice-only cases outside this runtime directory.
Similarity changes test priority only; it does not establish the current cause. Record every
version on which the failure and fix were actually verified.
