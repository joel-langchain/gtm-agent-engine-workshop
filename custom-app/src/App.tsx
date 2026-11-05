// Run eval: the dataset as a table with a Test button, inside LangSmith.
// Pick a dataset, press Test, and the app starts the team's eval pipeline
// through a short-lived sandbox, follows the run, and fills each row with the
// agent's answer, the judge's 1 to 5 score, and pass or fail. The eval itself
// runs in the pipeline with the team's own credentials.
import { useCallback, useEffect, useRef, useState } from 'react';
import { Badge } from '@langchain/macaw-components/Badge';
import { Banner } from '@langchain/macaw-components/Banner';
import { Button } from '@langchain/macaw-components/Button';
import { Card } from '@langchain/macaw-components/Card';
import { Select } from '@langchain/macaw-components/Select';
import { Spinner } from '@langchain/macaw-components/Spinner';
import { Text } from '@langchain/macaw-components/Text';
import { PlayIcon } from '@phosphor-icons/react/dist/ssr/Play';
import { ArrowSquareOutIcon } from '@phosphor-icons/react/dist/ssr/ArrowSquareOut';
import { CONFIG } from './config';
import {
  errText,
  experimentPath,
  experimentRuns,
  listDatasets,
  listExamples,
  listExperiments,
  runFeedback,
  sleep,
  type Dataset,
  type Example,
  type Experiment,
  type Feedback,
  type RunResult,
} from './lib/ls';
import { deleteSandbox, dispatchWorkflow, findRun, startSandbox, type Sandbox, type WorkflowRun } from './lib/pipeline';

type StepState = 'waiting' | 'active' | 'done' | 'failed';
interface Steps {
  sandbox: StepState;
  trigger: StepState;
  pipeline: StepState;
  experiment: StepState;
}
const IDLE: Steps = { sandbox: 'waiting', trigger: 'waiting', pipeline: 'waiting', experiment: 'waiting' };

interface RowResult {
  output: string;
  error: boolean;
  score: number | null;
  comment?: string | null;
}

const newRequestId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

function openInLangSmith(path: string) {
  const w = window.langsmith as unknown as { openUrl?: (u: string, o?: { newTab?: boolean }) => void };
  w.openUrl?.(path, { newTab: true });
}

/** The example's request text, whether it is stored as chat messages or a plain field. */
function inputText(inputs: Record<string, unknown>): string {
  const messages = inputs.messages as { content?: string }[] | undefined;
  if (Array.isArray(messages) && messages[0]?.content) return messages[0].content;
  const first = inputs.question ?? inputs.input ?? Object.values(inputs).find((v) => typeof v === 'string');
  return typeof first === 'string' ? first : JSON.stringify(inputs);
}

function expectedText(outputs?: Record<string, unknown> | null): string {
  if (!outputs) return '';
  const v = outputs.expected ?? outputs.expected_output ?? outputs.output ?? outputs.answer;
  return typeof v === 'string' ? v : JSON.stringify(outputs);
}

function agentName(ex: Example): string {
  const v = ex.inputs.agent ?? ex.metadata?.agent;
  return typeof v === 'string' ? v : '';
}

function outputText(outputs?: Record<string, unknown> | null): string {
  if (!outputs) return '';
  const v = outputs.output ?? outputs.answer;
  return typeof v === 'string' ? v : '';
}

/** Matches each experiment run to its dataset row, with the judge's score. */
async function loadResults(experimentId: string): Promise<Record<string, RowResult>> {
  const runs: RunResult[] = await experimentRuns(experimentId);
  const feedback: Feedback[] = await runFeedback(runs.map((r) => r.id));
  const byRun = new Map<string, Feedback>();
  for (const key of CONFIG.scoreKeys) {
    for (const f of feedback) if (f.key === key && !byRun.has(f.run_id)) byRun.set(f.run_id, f);
  }
  // The judge writes its explanation as a separate "reasoning" entry, with the text in value.
  const reasons = new Map<string, string>();
  for (const f of feedback) if (f.key === 'reasoning' && typeof f.value === 'string') reasons.set(f.run_id, f.value);
  const out: Record<string, RowResult> = {};
  for (const r of runs) {
    if (!r.reference_example_id) continue;
    const f = byRun.get(r.id);
    out[r.reference_example_id] = {
      output: outputText(r.outputs),
      error: Boolean(r.error) || r.status === 'error',
      score: f?.score ?? null,
      comment: reasons.get(r.id) ?? f?.comment,
    };
  }
  return out;
}

export function App({ metadata }: { data: unknown; metadata: RenderMetadata }) {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [datasetId, setDatasetId] = useState<string>();
  const [examples, setExamples] = useState<Example[] | null>(null);
  const [history, setHistory] = useState<Experiment[]>([]);
  const [shown, setShown] = useState<Experiment | null>(null);
  const [results, setResults] = useState<Record<string, RowResult>>({});
  const [steps, setSteps] = useState<Steps>(IDLE);
  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sandboxRef = useRef<Sandbox | null>(null);
  const pollRef = useRef(0);

  const dataset = datasets?.find((d) => d.id === datasetId);
  const workspaceId = metadata.workspaceId ?? dataset?.tenant_id ?? '';

  useEffect(() => {
    listDatasets(CONFIG.defaultDataset).then(
      (list) => {
        setDatasets(list);
        setDatasetId((list.find((d) => d.name === CONFIG.defaultDataset) ?? list[0])?.id);
      },
      (err) => setError(errText(err))
    );
  }, []);

  /** Shows an experiment's results in the table, and keeps checking until the judge has scored every row. */
  const show = useCallback(async (experiment: Experiment) => {
    const token = ++pollRef.current;
    setShown(experiment);
    for (let i = 0; i < 40 && token === pollRef.current; i++) {
      const res = await loadResults(experiment.id).catch(() => null);
      if (token !== pollRef.current) return;
      if (res) {
        setResults(res);
        const rows = Object.values(res);
        if (rows.length > 0 && rows.every((r) => r.error || r.score !== null)) return;
      }
      await sleep(8000);
    }
  }, []);

  const loadDataset = useCallback(
    async (id: string) => {
      pollRef.current++;
      setExamples(null);
      setResults({});
      setShown(null);
      const [ex, exps] = await Promise.all([listExamples(id), listExperiments(id, 20)]);
      setExamples(ex);
      const fromApp = exps.filter((e) => e.name.startsWith('ui-')).slice(0, 5);
      setHistory(fromApp);
      if (fromApp[0]) show(fromApp[0]);
    },
    [show]
  );

  useEffect(() => {
    if (datasetId) loadDataset(datasetId).catch((err) => setError(errText(err)));
  }, [datasetId, loadDataset]);

  const mark = (patch: Partial<Steps>) => setSteps((s) => ({ ...s, ...patch }));

  const start = async () => {
    if (!dataset) return;
    const requestId = newRequestId();
    pollRef.current++;
    setBusy(true);
    setError(null);
    setRun(null);
    setResults({});
    setShown(null);
    setSteps({ ...IDLE, sandbox: 'active' });
    let failedAt: keyof Steps = 'sandbox';
    let found: Experiment | undefined;
    try {
      const sandbox = await startSandbox(requestId);
      sandboxRef.current = sandbox;
      mark({ sandbox: 'done', trigger: 'active' });

      failedAt = 'trigger';
      await dispatchWorkflow(sandbox, dataset.name, requestId, workspaceId);
      mark({ trigger: 'done', pipeline: 'active' });

      failedAt = 'pipeline';
      const deadline = Date.now() + CONFIG.timeoutMs;
      let current: WorkflowRun | null = null;
      while (Date.now() < deadline) {
        await sleep(CONFIG.pollMs);
        current = await findRun(sandbox, requestId);
        if (current) setRun(current);
        if (current?.status === 'completed') break;
      }
      if (current?.status !== 'completed') throw new Error('The pipeline did not finish within 15 minutes.');
      if (current.conclusion !== 'success') throw new Error(`The pipeline finished with "${current.conclusion}". Open the GitHub run for the log.`);
      mark({ pipeline: 'done', experiment: 'active' });

      failedAt = 'experiment';
      for (let i = 0; i < 6 && !found; i++) {
        const list = await listExperiments(dataset.id, 20);
        found = list.find((e) => e.name.startsWith(`ui-${requestId}`));
        if (!found) await sleep(3000);
      }
      if (!found) throw new Error('The pipeline finished but no experiment with this run’s name was found on the dataset.');
      mark({ experiment: 'done' });
    } catch (err) {
      mark({ [failedAt]: 'failed' } as Partial<Steps>);
      setError(errText(err));
    } finally {
      await deleteSandbox(sandboxRef.current);
      sandboxRef.current = null;
      setBusy(false);
    }
    if (found) {
      listExperiments(dataset.id, 20)
        .then((exps) => setHistory(exps.filter((e) => e.name.startsWith('ui-')).slice(0, 5)))
        .catch(() => undefined);
      show(found);
    }
  };

  const rows = examples ?? [];
  const scored = rows.map((ex) => results[ex.id]).filter((r): r is RowResult => Boolean(r));
  const passed = scored.filter((r) => !r.error && r.score !== null && r.score >= CONFIG.passThreshold).length;
  const judged = scored.filter((r) => !r.error && r.score !== null);
  const errored = scored.filter((r) => r.error).length;
  const average = judged.length ? judged.reduce((s, r) => s + (r.score as number), 0) / judged.length : null;
  const waitingForJudge = scored.some((r) => !r.error && r.score === null);

  return (
    <div className="min-h-screen bg-surface-level-1">
      <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-space-5 p-space-6">
        <header className="flex flex-col gap-space-1">
          <Text variant="h3" as="h1" weight="semibold">
            Run eval
          </Text>
          <Text variant="sm" color="secondary" className="max-w-[80ch]">
            Pick a dataset and press Test. The team&apos;s pipeline calls each row&apos;s agent with its own credentials, and the judge
            on the dataset scores each answer from 1 to 5 against the expected output.
          </Text>
        </header>

        <Card className="flex flex-col gap-space-3 p-space-5">
          <div className="flex flex-wrap items-end gap-space-3">
            <div className="flex min-w-[280px] flex-1 flex-col gap-space-1">
              <Text variant="sm" weight="medium">
                Dataset
              </Text>
              {datasets ? (
                <Select
                  value={datasetId}
                  onChange={(v) => v && setDatasetId(v)}
                  disabled={busy}
                  placeholder="Choose a dataset"
                  options={datasets.map((d) => ({
                    value: d.id,
                    label: d.example_count != null ? `${d.name} (${d.example_count} examples)` : d.name,
                  }))}
                />
              ) : (
                <Spinner size="sm" />
              )}
            </div>
            <Button leftDecorator={PlayIcon} onClick={start} disabled={!dataset || busy}>
              {busy ? 'Running' : 'Test'}
            </Button>
          </div>
          <Text variant="xs" color="tertiary">
            {`Pipeline: ${CONFIG.repo}, workflow ${CONFIG.workflow} on ${CONFIG.ref}. Pass at ${CONFIG.passThreshold} or above.`}
          </Text>
          {steps.sandbox !== 'waiting' && (
            <div className="flex flex-wrap items-center gap-space-4 border-t border-subtle pt-space-3">
              <Step state={steps.sandbox} label="Sandbox" />
              <Step state={steps.trigger} label="Trigger pipeline" />
              <Step state={steps.pipeline} label={run ? `Pipeline ${run.status.replace('_', ' ')}` : 'Pipeline'} />
              <Step state={steps.experiment} label="Experiment" />
              {run && (
                <Text variant="xs" color="tertiary" className="break-all">
                  {run.html_url}
                </Text>
              )}
            </div>
          )}
        </Card>

        {error && (
          <Banner intent="error" title="The run stopped">
            {error}
          </Banner>
        )}

        {shown && dataset && (
          <div className="flex flex-wrap items-center justify-between gap-space-3">
            <div className="flex flex-wrap items-center gap-space-3">
              <Text variant="md" weight="semibold">
                {judged.length && !waitingForJudge ? `Pass rate ${Math.round((passed / rows.length) * 100)}%` : 'Results'}
              </Text>
              {judged.length > 0 && !waitingForJudge && (
                <Badge size="sm" color="primary">{`${passed} of ${rows.length} passed`}</Badge>
              )}
              {average !== null && !waitingForJudge && <Badge size="sm" color="secondary">{`Average ${average.toFixed(1)} / 5`}</Badge>}
              {errored > 0 && <Badge size="sm" color="warning">{`${errored} errored`}</Badge>}
              {waitingForJudge && (
                <span className="flex items-center gap-space-2">
                  <Spinner size="sm" />
                  <Text variant="xs" color="tertiary">
                    Judge scoring
                  </Text>
                </span>
              )}
              <Text variant="xs" color="tertiary">
                {shown.name}
              </Text>
            </div>
            <Button
              size="sm"
              variant="outlined"
              color="secondary"
              rightDecorator={ArrowSquareOutIcon}
              onClick={() => openInLangSmith(experimentPath(workspaceId, dataset.id, shown.id))}
            >
              Open experiment
            </Button>
          </div>
        )}

        <Card className="overflow-x-auto p-0">
          {examples === null ? (
            <div className="p-space-5">
              <Spinner size="sm" />
            </div>
          ) : (
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-default">
                  {['#', 'Input', 'Expected output', 'Agent', 'Output', 'Score and reason', 'Result'].map((h) => (
                    <th key={h} className="px-space-3 py-space-2 align-bottom">
                      <Text variant="xs" color="tertiary" weight="medium">
                        {h}
                      </Text>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((ex, i) => {
                  const r = results[ex.id];
                  return (
                    <tr key={ex.id} className="border-b border-subtle align-top last:border-b-0">
                      <td className="px-space-3 py-space-3">
                        <Text variant="xs" color="tertiary">
                          {i + 1}
                        </Text>
                      </td>
                      <Cell text={inputText(ex.inputs)} width="w-[22%]" />
                      <Cell text={expectedText(ex.outputs)} width="w-[22%]" />
                      <td className="px-space-3 py-space-3">
                        <Text variant="sm">{agentName(ex) || '–'}</Text>
                      </td>
                      <Cell text={r ? (r.error ? 'The agent errored on this example.' : r.output) : '–'} width="w-[26%]" muted={!r} />
                      <td className="w-[20%] px-space-3 py-space-3">
                        <Text variant="sm" weight="medium">
                          {r && r.score !== null ? `${r.score} / 5` : '–'}
                        </Text>
                        {r?.comment && (
                          <Text variant="xs" color="tertiary" className="line-clamp-4" title={r.comment}>
                            {r.comment}
                          </Text>
                        )}
                      </td>
                      <td className="px-space-3 py-space-3">
                        <ResultBadge result={r} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Card>

        {history.length > 0 && dataset && (
          <Card className="flex flex-col gap-space-2 p-space-5">
            <Text variant="md" as="h2" weight="semibold">
              Earlier runs from this app
            </Text>
            {history.map((e) => (
              <div key={e.id} className="flex items-center justify-between gap-space-3">
                <div className="flex flex-col">
                  <Text variant="sm" weight="medium">
                    {e.name}
                  </Text>
                  <Text variant="xs" color="tertiary">
                    {new Date(e.start_time).toLocaleString()}
                  </Text>
                </div>
                <Button size="sm" variant="outlined" color="secondary" disabled={busy} onClick={() => show(e)}>
                  Show in table
                </Button>
              </div>
            ))}
          </Card>
        )}
      </div>
    </div>
  );
}

function Cell({ text, width, muted }: { text: string; width: string; muted?: boolean }) {
  return (
    <td className={`px-space-3 py-space-3 ${width}`}>
      <Text variant="sm" color={muted ? 'tertiary' : undefined} className="line-clamp-3" title={text}>
        {text}
      </Text>
    </td>
  );
}

function ResultBadge({ result }: { result?: RowResult }) {
  if (!result) return <Text variant="sm" color="tertiary">{'–'}</Text>;
  if (result.error)
    return (
      <Badge size="sm" color="warning">
        Error
      </Badge>
    );
  if (result.score === null) return <Spinner size="sm" />;
  return result.score >= CONFIG.passThreshold ? (
    <Badge size="sm" color="success">
      Pass
    </Badge>
  ) : (
    <Badge size="sm" color="error">
      Fail
    </Badge>
  );
}

function Step({ state, label }: { state: StepState; label: string }) {
  const indicator =
    state === 'done' ? (
      <Badge size="sm" color="success">
        Done
      </Badge>
    ) : state === 'failed' ? (
      <Badge size="sm" color="error">
        Failed
      </Badge>
    ) : state === 'active' ? (
      <Spinner size="sm" />
    ) : (
      <Badge size="sm" color="secondary">
        Waiting
      </Badge>
    );
  return (
    <span className="flex items-center gap-space-2">
      {indicator}
      <Text variant="sm">{label}</Text>
    </span>
  );
}
