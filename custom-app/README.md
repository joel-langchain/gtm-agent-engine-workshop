# Run eval custom app

A [LangSmith custom app](https://docs.langchain.com/langsmith/custom-apps) that runs an evaluation from a button. Pick a dataset and press **Test**. The app starts this repository's [`run-eval.yml`](../.github/workflows/run-eval.yml) workflow, then shows each row's input, expected output, agent output, the judge's 1 to 5 score with its reason, and pass or fail, with the pass rate on top.

## How it works

A custom app runs inside LangSmith and has no network access of its own. Every call it makes goes to the LangSmith API with the viewer's own permissions, so people only see the datasets and experiments they can already open.

To reach GitHub, the app starts a short-lived [LangSmith sandbox](https://docs.langchain.com/langsmith/sandboxes) and runs `curl` inside it. The sandbox's auth proxy adds the GitHub token from a workspace secret to requests for `api.github.com`, so the token never appears in the app or in the sandbox code.

1. `src/App.tsx` lists the datasets, starts a run, and draws the results table.
2. `src/lib/pipeline.ts` starts the sandbox, triggers the workflow with the dataset name and a request ID, finds the run by that ID, and deletes the sandbox at the end.
3. The workflow runs `eval.py` on the dataset, which creates an experiment named `ui-<request ID>`.
4. `src/lib/ls.ts` reads the experiment's runs and the judge's feedback for the table.

`src/config.ts` holds the repository, workflow, default dataset, feedback keys, and pass threshold.

## Setup

1. Create a GitHub token with Actions read and write on the repository, and save it in LangSmith as the workspace secret `GITHUB_ACTIONS_TOKEN`.
2. Keep `run-eval.yml` on the default branch, since GitHub only dispatches workflows from there. The secrets it reads are listed in the workflow file.
3. Attach an LLM-as-judge evaluator to the dataset that writes a `score` feedback from 1 to 5 with its reasoning.
4. Give users permission to create and run sandboxes.
5. Build and push the app:

```bash
npm install
npm run build
langsmith apps push --workspace <workspace-id> --name "Run eval"
```

`langsmith apps dev` runs it locally against your workspace first.

## Using GitLab instead of GitHub

Only `src/lib/pipeline.ts` and `src/config.ts` change.

- In the sandbox's proxy rule, match your GitLab host instead of `api.github.com` and point the header at a workspace secret holding a GitLab token. GitLab accepts `Authorization: Bearer <token>`, so the header shape stays the same.
- Replace the dispatch call with [`POST /api/v4/projects/:id/pipeline`](https://docs.gitlab.com/api/pipelines/), with `ref` and the dataset name and request ID as pipeline variables, or as `inputs` if the CI file declares them under `spec:inputs`.
- GitLab returns the pipeline `id` and `web_url` straight away, so `findRun` becomes `GET /api/v4/projects/:id/pipelines/:pipeline_id` and no title matching is needed. A finished pipeline's `status` is `success`, `failed`, or `canceled`.
- The sandbox needs to be able to reach the GitLab host.

The GitLab CI job does what `run-eval.yml` does: install dependencies, then run the eval script with the dataset name and experiment prefix.
