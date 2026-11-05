// Everything the app needs to know about the pipeline it starts. The GitHub
// token never appears here: it is a workspace secret that the sandbox's auth
// proxy adds to requests to api.github.com.
export const CONFIG = {
  repo: 'joel-langchain/gtm-agent-engine-workshop',
  workflow: 'run-eval.yml',
  ref: 'main',
  /** Workspace secret holding a GitHub token with Actions read and write on the repo. */
  tokenSecret: 'GITHUB_ACTIONS_TOKEN',
  /** Send the demo agent's model calls through the LangSmith LLM Gateway, so the
   *  pipeline authenticates with its LangSmith key and needs no separate model key. */
  useGateway: true,
  /** Preselected in the dataset picker when it exists. */
  defaultDataset: 'gtm-agent-test-suite',
  /** Judge feedback keys to read as the 1 to 5 score, in order of preference. */
  scoreKeys: ['answer_quality', 'score'],
  /** A row passes at this score or above, as in the team's current tool. */
  passThreshold: 4,
  /** How often to check the pipeline, and how long to wait before giving up. */
  pollMs: 5000,
  timeoutMs: 15 * 60 * 1000,
};
