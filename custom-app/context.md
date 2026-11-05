# Run eval: context

## What it is

A button in LangSmith that runs an evaluation for people who don't run code.
The agent under test runs outside LangSmith, so experiments come from a script
in CI. The app starts that CI pipeline and shows the results in a table. See
README.md for how it fits together and how to set it up.

## Decisions

- The app never holds a token. The sandbox's auth proxy adds it from the
  `GITHUB_ACTIONS_TOKEN` workspace secret.
- The workflow is matched by the request ID in its `run-name`, because GitHub's
  dispatch call returns no run ID.
- `useGateway` in `src/config.ts` sends the demo agent's model calls through the
  LangSmith LLM Gateway, so the workflow authenticates with its LangSmith key
  and needs no separate model key. The gateway needs `provider/model` names,
  which is why the workflow sets `GTM_MODEL=openai/gpt-4o-mini`.
- The pass rate is hidden until the judge has scored every row, so a partly
  scored run doesn't show a misleading figure.
- The dataset picker lists the first 100 datasets and also fetches
  `defaultDataset` by name, so the default is there in larger workspaces.

## Notes

- `window.langsmith.openUrl` only opens LangSmith URLs, so the GitHub run link
  is shown as text.
- Sandbox endpoints are `/api/v2/sandboxes/...` and go through
  `window.langsmith.call` like any other API call.
- A run of six rows took about 100 seconds in the pipeline, and the judge
  scored it about two minutes later.
- The workflow's API key must be able to write to the workspace the app runs
  in, or the run succeeds but the app can't find the experiment.
