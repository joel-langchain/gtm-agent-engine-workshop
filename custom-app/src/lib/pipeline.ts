// Starts the GitHub Actions eval workflow from inside LangSmith. The app has no
// network of its own, so it borrows a short-lived LangSmith sandbox: the
// sandbox runs curl against api.github.com, and the sandbox's auth proxy adds
// the GitHub token from a workspace secret. Nothing secret passes through the app.
import { CONFIG } from '../config';
import { call, sleep } from './ls';

interface ExecResult {
  stdout: string;
  stderr: string;
  exit_code: number;
}

export interface Sandbox {
  id: string;
  name: string;
}

export interface WorkflowRun {
  id: number;
  html_url: string;
  status: string; // queued | in_progress | completed
  conclusion: string | null; // success | failure | cancelled | ...
  display_title: string;
}

const GITHUB_API = `https://api.github.com/repos/${CONFIG.repo}/actions/workflows/${CONFIG.workflow}`;
const GITHUB_HEADERS = `-H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2022-11-28'`;

export async function startSandbox(requestId: string): Promise<Sandbox> {
  const name = `eval-trigger-${requestId}`;
  const box = await call<{ id: string; name: string; status: string }>('POST /api/v2/sandboxes/boxes', {
    body: {
      name,
      persistent: false,
      idle_ttl_seconds: 600,
      labels: { purpose: 'eval-run-button' },
      proxy_config: {
        rules: [
          {
            name: 'github-api',
            match_hosts: ['api.github.com'],
            headers: [{ name: 'Authorization', type: 'workspace_secret', value: `Bearer {${CONFIG.tokenSecret}}` }],
          },
        ],
      },
    },
  });
  for (let i = 0; i < 30 && box.status !== 'ready'; i++) {
    await sleep(1000);
    const s = await call<{ status: string }>(`GET /api/v2/sandboxes/boxes/${name}/status`);
    box.status = s.status;
  }
  if (box.status !== 'ready') throw new Error(`Sandbox did not start (status: ${box.status})`);
  return { id: box.id, name };
}

export async function deleteSandbox(sandbox: Sandbox | null) {
  if (!sandbox) return;
  try {
    await call(`DELETE /api/v2/sandboxes/boxes/${sandbox.name}`);
  } catch {
    // Non-persistent with an idle timeout, so it is cleaned up regardless.
  }
}

async function exec(sandbox: Sandbox, command: string, env?: Record<string, string>): Promise<ExecResult> {
  return call<ExecResult>(`POST /api/v2/sandboxes/${sandbox.id}/execute`, {
    body: { command, timeout_seconds: 60, ...(env ? { run_config: { env_vars: env } } : {}) },
  });
}

/** Starts the workflow on main. GitHub answers 204 with no body. */
export async function dispatchWorkflow(sandbox: Sandbox, datasetName: string, requestId: string, workspaceId: string) {
  const payload = JSON.stringify({
    ref: CONFIG.ref,
    inputs: {
      dataset_name: datasetName,
      experiment_prefix: `ui-${requestId}`,
      request_id: requestId,
      workspace_id: workspaceId,
      use_gateway: CONFIG.useGateway ? 'true' : 'false',
    },
  });
  const res = await exec(
    sandbox,
    `curl -sS -o /tmp/out -w '%{http_code}' -X POST ${GITHUB_HEADERS} ${GITHUB_API}/dispatches --data "$PAYLOAD"; echo; cat /tmp/out`,
    { PAYLOAD: payload }
  );
  const [code, ...body] = res.stdout.split('\n');
  if (!code.startsWith('2')) {
    throw new Error(`GitHub refused the trigger (HTTP ${code || '?'}): ${body.join(' ').slice(0, 300) || res.stderr}`);
  }
}

/** The workflow run started by this request, matched on the request ID in its title. */
export async function findRun(sandbox: Sandbox, requestId: string): Promise<WorkflowRun | null> {
  const res = await exec(sandbox, `curl -sS ${GITHUB_HEADERS} '${GITHUB_API}/runs?event=workflow_dispatch&per_page=20'`);
  let parsed: { workflow_runs?: WorkflowRun[]; message?: string };
  try {
    parsed = JSON.parse(res.stdout);
  } catch {
    throw new Error(`Unexpected reply from GitHub: ${res.stdout.slice(0, 200) || res.stderr}`);
  }
  if (!parsed.workflow_runs) throw new Error(`GitHub: ${parsed.message ?? 'no runs returned'}`);
  return parsed.workflow_runs.find((r) => r.display_title.includes(requestId)) ?? null;
}
