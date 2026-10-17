import { readFile } from 'node:fs/promises';
import { parse } from 'yaml';
import { z } from 'zod';
import { contractSchema, experimentSchema, featureSchema } from './types.js';

export async function yamlFile(path: string): Promise<unknown> {
  return parse(await readFile(path, 'utf8'));
}
export async function loadConfig(experimentPath: string, contractPath: string, featurePath = 'configs/features.yaml') {
  const [experiment, contract, features] = await Promise.all([
    yamlFile(experimentPath), yamlFile(contractPath), yamlFile(featurePath),
  ]);
  const parsedFeatures = z.array(featureSchema).min(1).parse(features);
  if (new Set(parsedFeatures.map((f) => f.id)).size !== parsedFeatures.length) throw new Error('Duplicate feature IDs');
  if (!parsedFeatures.some((f) => f.group === 'domestic') || !parsedFeatures.some((f) => f.group === 'external')) {
    throw new Error('Both domestic and external feature groups are required');
  }
  return { experiment: experimentSchema.parse(experiment), contract: contractSchema.parse(contract), features: parsedFeatures };
}
