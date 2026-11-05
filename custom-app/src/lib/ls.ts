// LangSmith API access, entirely through window.langsmith.call (the sandbox has no
// network of its own).

type Params = Record<string, string | string[]>;

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** One API call, retried with backoff on rate limits. */
export async function call<T>(operation: string, args?: { params?: Params; body?: unknown }): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return (await window.langsmith.call(operation, args)) as T;
    } catch (err) {
      const msg = errText(err);
      if (attempt < 4 && /429|rate|too many/i.test(msg)) {
        await sleep(1500 * 2 ** attempt + Math.random() * 500);
        continue;
      }
      throw err;
    }
  }
}

export const errText = (err: unknown) => String((err as Error)?.message ?? err);

export interface Dataset {
  id: string;
  name: string;
  tenant_id: string;
  example_count?: number;
}

export interface Experiment {
  id: string;
  name: string;
  start_time: string;
  run_count?: number;
  /** Share of the experiment's runs that errored, 0 to 1. */
  error_rate?: number | null;
  feedback_stats?: Record<string, { avg?: number | null; n?: number }> | null;
}

/** Datasets for the picker. Busy workspaces have more than one page, so the
 *  default dataset is fetched by name and always included. */
export async function listDatasets(defaultName: string): Promise<Dataset[]> {
  const [list, byName] = await Promise.all([
    call<Dataset[]>('GET /api/v1/datasets', { params: { limit: '100' } }),
    call<Dataset[]>('GET /api/v1/datasets', { params: { name: defaultName } }).catch(() => [] as Dataset[]),
  ]);
  const merged = new Map([...byName, ...list].map((d) => [d.id, d]));
  return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/** Most recent experiments on a dataset, newest first, with feedback averages. */
export async function listExperiments(datasetId: string, limit = 10): Promise<Experiment[]> {
  return call<Experiment[]>('GET /api/v1/sessions', {
    params: {
      reference_dataset: datasetId,
      sort_by: 'start_time',
      sort_by_desc: 'true',
      include_stats: 'true',
      limit: String(limit),
    },
  });
}

export interface Example {
  id: string;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
}

export interface RunResult {
  id: string;
  reference_example_id: string | null;
  status?: string;
  outputs?: Record<string, unknown> | null;
  error?: string | null;
}

export interface Feedback {
  run_id: string;
  key: string;
  score: number | null;
  value?: unknown;
  comment?: string | null;
}

export async function listExamples(datasetId: string): Promise<Example[]> {
  return call<Example[]>('GET /api/v1/examples', { params: { dataset: datasetId, limit: '100' } });
}

/** Root runs of an experiment, one per dataset example. */
export async function experimentRuns(experimentId: string): Promise<RunResult[]> {
  const res = await call<{ items: RunResult[] }>('POST /v2/runs/query', {
    body: {
      project_ids: [experimentId],
      is_root: true,
      page_size: 100,
      selects: ['ID', 'OUTPUTS', 'ERROR', 'REFERENCE_EXAMPLE_ID', 'STATUS'],
    },
  });
  return res.items ?? [];
}

export async function runFeedback(runIds: string[]): Promise<Feedback[]> {
  if (runIds.length === 0) return [];
  return call<Feedback[]>('GET /api/v1/feedback', { params: { run: runIds, limit: '100' } });
}

export function experimentPath(workspaceId: string, datasetId: string, experimentId: string) {
  return `/o/${workspaceId}/datasets/${datasetId}/compare?selectedSessions=${experimentId}`;
}
