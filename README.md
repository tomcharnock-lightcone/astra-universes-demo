# ASTRA universes demo

A small runnable demo of the universes proposal for [ASTRA](https://astra-spec.org):
universes declared in `astra.yaml` with one written as a diff of another and an
output that compares the two through pins.

It runs the extended iris example from
[RFC-0003](https://github.com/LightconeResearch/astra-spec/pull/65) as the
universes proposal reworks it. The code it runs against is on the
`proto/universes` branches of
[astra-spec](https://github.com/LightconeResearch/astra-spec/tree/proto/universes)
and [astra-tools](https://github.com/LightconeResearch/astra-tools/tree/proto/universes).

## What it shows

`iris/astra.yaml` declares two universes. `svm_standard` is a diff of
`rf_standard` - it only switches the model and inherits the rest:

```yaml
universes:
  - id: rf_standard
    decisions:
      scaling: standard
      model: random_forest
      test_size: small
      random_seed: seed_42
  - id: svm_standard
    derived_from: rf_standard
    decisions:
      model: svm
```

`f1_score` depends on the decisions so it's built once per universe.
`model_comparison` reads `f1_score#rf_standard` and `f1_score#svm_standard` -
two pins - and declares no decisions of its own so it's built once:

```
results/
  model_comparison.json
  rf_standard/
    trained_output.pkl
    f1_score.json
  svm_standard/
    trained_output.pkl
    f1_score.json
```

Where results go is provisional - it's for a later layout proposal
([#72](https://github.com/LightconeResearch/astra-spec/issues/72)).

## Running it

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- `python run.py iris` builds everything with stub scripts;
- `python run.py iris-sklearn` builds the same thing with real scikit-learn
  models (it takes a second or two);
- `python run.py iris model_comparison` builds only the comparison and what it
  needs;
- `python run.py iris 'f1_score#svm_standard'` builds one output in one
  universe - the same pin syntax as in `astra.yaml`;
- `python run.py iris --dry-run` prints the commands without running them;
- `pytest` runs the checks in `test_demo.py`.

A result that's already on disk is skipped. Pass `--force` to rebuild it.

## The two projects

`iris/` and `iris-sklearn/` share the same `astra.yaml`. In `iris/` the scripts
are stubs that write down what they were given - the decisions, the inputs they
read and what those contained - so `results/model_comparison.json` shows it drew
on both universes. `iris-sklearn/` trains and scores real models. With the
random forest and the SVM above it reports a macro-F1 of about 0.90 and 0.97
respectively.

## How the runner works

`run.py` is deliberately small. It validates the analysis with astra-tools and
asks `astra.resolve.plan` for every output to build - in which universe, where
its result goes and which files it reads. Then it runs each recipe in
dependency order with its placeholders filled in by
`astra.resolve.render_command`.
Recipes run from the project folder with the runner's own Python as `python`.
