import { sha256, write, writeJson } from './io.js';

// Raw provider downloads deliberately are NOT normalized into invented release/vintage timestamps.
// A reviewed timing sidecar and series contract are required before the research loader accepts them.
export async function downloadProvider(options: {
  provider: 'fred' | 'ecos'; series: string; start: string; end: string; item?: string; output: string;
}) {
  const key = process.env[options.provider === 'fred' ? 'FRED_API_KEY' : 'ECOS_API_KEY'];
  if (!key) throw new Error(`Missing ${options.provider.toUpperCase()}_API_KEY`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(options.start) || !/^\d{4}-\d{2}-\d{2}$/.test(options.end) || options.start > options.end) {
    throw new Error('Use an ordered YYYY-MM-DD date range');
  }
  const requests: { url: string; offset: number }[] = [];
  if (options.provider === 'fred') {
    const url = new URL('https://api.stlouisfed.org/fred/series/observations');
    url.search = new URLSearchParams({ api_key: key, series_id: options.series, file_type: 'json',
      observation_start: options.start, observation_end: options.end, limit: '100000' }).toString();
    requests.push({ url: url.toString(), offset: 0 });
  } else {
    if (!options.item) throw new Error('ECOS requires --item with the verified statistic item code');
    requests.push({ url: '', offset: 1 });
  }
  const pages: unknown[] = [];
  let count = 0;
  for (let page = 0; page < requests.length; page++) {
    if (page > 1000) throw new Error('Provider pagination exceeded safety limit');
    const request = requests[page];
    const url = options.provider === 'fred' ? request.url
      : `https://ecos.bok.or.kr/api/StatisticSearch/${encodeURIComponent(key)}/json/kr/${request.offset}/${request.offset + 999}/${encodeURIComponent(options.series)}/M/${options.start.slice(0, 7).replace('-', '')}/${options.end.slice(0, 7).replace('-', '')}/${options.item!.split(',').map(encodeURIComponent).join('/')}`;
    let response: Response;
    try { response = await fetch(url, { signal: AbortSignal.timeout(30_000) }); }
    catch { throw new Error('Provider request failed (URL omitted to protect API key)'); }
    if (!response.ok) throw new Error(`Provider HTTP ${response.status} (URL omitted)`);
    const body = await response.text();
    let json: Record<string, any>;
    try { json = JSON.parse(body); } catch { throw new Error('Provider returned non-JSON response'); }
    if (json.error_code || json.RESULT) throw new Error('Provider rejected the request; verify key and series codes');
    const rows = options.provider === 'fred' ? json.observations : json.StatisticSearch?.row;
    if (!Array.isArray(rows)) throw new Error('Unexpected provider response: missing observations');
    count += rows.length;
    pages.push(json);
    const total = Number(options.provider === 'fred' ? json.count : json.StatisticSearch?.list_total_count);
    const size = options.provider === 'fred' ? 100000 : 1000;
    if (count < total) {
      if (options.provider === 'fred') {
        const next = new URL(url); next.searchParams.set('offset', String(count));
        requests.push({ url: next.toString(), offset: count });
      } else requests.push({ url: '', offset: request.offset + size });
    }
  }
  const content = JSON.stringify(pages, null, 2) + '\n';
  await write(options.output, content);
  await writeJson(options.output + '.metadata.json', {
    provider: options.provider, series_id: options.series, item: options.item ?? null,
    requested_range: [options.start, options.end], retrieved_at: new Date().toISOString(),
    sha256: sha256(content), observations: count, pages: pages.length,
    timing_status: 'UNVERIFIED: no inferred historical release or availability timestamps',
    source: options.provider === 'fred' ? 'https://fred.stlouisfed.org/docs/api/fred/' : 'https://ecos.bok.or.kr/api/',
  });
  return { observations: count, output: options.output };
}
