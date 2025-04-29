/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Genkit, z } from 'genkit';
import { defineAction } from '@genkit-ai/core';
import {
  CommonRetrieverOptionsSchema,
  indexerRef,
  retrieverRef,
  Document
} from 'genkit/retriever';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import { EmbedderArgument } from 'genkit/embedder';

import { v4 as uuidv4 } from 'uuid';
import { PostgresEngine } from './engine';

const PostgresRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
  filter: z.record(z.string(), z.any()).optional(),
});

const PostgresIndexerOptionsSchema = z.object({});

/**
 * postgresRetrieverRef function creates a retriever for Postgres.
 * @param params The params for the new Postgres retriever
 * @param params.tableName The table name for the postgres retriever
If not specified, the default label will be `Postgres - <tableName>`
 * @returns A reference to a Postgres retriever.
 */
export const postgresRetrieverRef = (params: {
  tableName: string;
}) => {
  return retrieverRef({
    name: `postgres/${params.tableName}`,
    info: {
      label: params.tableName ?? `Postgres - ${params.tableName}`,
    },
    configSchema: PostgresRetrieverOptionsSchema,
  });
};

/**
 * postgresIndexerRef function creates an indexer for Postgres.
 * @param params The params for the new Postgres indexer.
 * @param params.tableName The table name for the Postgres indexer.
If not specified, the default label will be `Postgres - <tableName>`
 * @returns A reference to a Postgres indexer.
 */
export const postgresIndexerRef = (params: {
  tableName: string;
}) => {
  return indexerRef({
    name: `postgres/${params.tableName}`,
    info: {
      label: params.tableName ?? `Postgres - ${params.tableName}`,
    },
    configSchema: PostgresIndexerOptionsSchema.optional(),
  });
};

// Types
type PostgresIndexerConfig<EmbedderCustomOptions extends z.ZodTypeAny> = {
  tableName: string;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
  engine: PostgresEngine;
};

type IndexedDocument = {
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
};

// Constants
const CHUNK_SIZE = 100;
const DEFAULT_SCHEMA = 'public';
const DEFAULT_CONTENT_COLUMN = 'content';
const DEFAULT_EMBEDDING_COLUMN = 'embedding';
const DEFAULT_ID_COLUMN = 'id';
const DEFAULT_DISTANCE_STRATEGY = 'cosine';

/**
 * Configures a Postgres indexer with the given parameters
 */
function configurePostgresIndexer<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: PostgresIndexerConfig<EmbedderCustomOptions> & {
    schemaName?: string;
    contentColumn?: string;
    embeddingColumn?: string;
    metadataColumns?: string[];
    idColumn?: string;
    metadataJsonColumn?: string;
    distanceStrategy?: 'cosine' | 'ip' | 'l2';
  }
) {
  const {
    tableName,
    engine,
    schemaName = DEFAULT_SCHEMA,
    contentColumn = DEFAULT_CONTENT_COLUMN,
    embeddingColumn = DEFAULT_EMBEDDING_COLUMN,
    metadataColumns = [],
    idColumn = DEFAULT_ID_COLUMN,
    metadataJsonColumn,
    distanceStrategy = DEFAULT_DISTANCE_STRATEGY,
    embedder: embedderName = 'text-embedding-004'
  } = params;

  /**
   * Generates embeddings for a batch of documents
   */
  async function generateEmbeddings(documents: Document[]): Promise<IndexedDocument[]> {
    const results: IndexedDocument[] = [];

    for (let i = 0; i < documents.length; i += CHUNK_SIZE) {
      const chunk = documents.slice(i, i + CHUNK_SIZE);
      const batchDocuments = chunk.map(doc => new Document({ content: doc.content }));

      let batchEmbeddings: number[][] = [];

      if (typeof embedderName === 'function') {
        batchEmbeddings = await Promise.all(
          batchDocuments.map(doc => 
            (embedderName as unknown as (input: { input: { content: { text: string }[] }[] }) => Promise<{ embedding: number[] }[]>)({ 
              input: [{ content: [{ text: String(doc.content) }] }] 
            }).then(result => result[0]?.embedding || [])
          )
        );
      } else {
        const embedResult = await ai.embed({
          embedder: embedderName,
          content: batchDocuments.map(doc => String(doc.content)).join(' '),
        });
        batchEmbeddings = embedResult.map(r => r.embedding);
      }

      const chunkResults = chunk.map((doc, index) => ({
        content: String(doc.content),
        embedding: batchEmbeddings[index],
        metadata: doc.metadata || {},
      }));

      results.push(...chunkResults);
    }

    return results;
  }

  /**
   * Adds documents to the index
   */
  async function addDocuments(
    documents: Document[],
    options?: { ids?: string[]; batchSize?: number }
  ) {
    if (!engine) {
      throw new Error('Indexer not initialized. Call initialize() first.');
    }

    const vectors = await generateEmbeddings(documents);
    const batchSize = options?.batchSize || CHUNK_SIZE;
    const ids: string[] = options?.ids || documents.map(doc => 
      (doc as { id?: string }).id || uuidv4()
    );

    // Process documents in batches
    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize);
      const batchIds = ids.slice(i, i + batchSize);

      await engine.pool.raw(
        `
        INSERT INTO "${schemaName}"."${tableName}"(
          "${idColumn}",
          "${contentColumn}",
          "${embeddingColumn}",
          ${metadataColumns.map(col => `"${col}"`).join(', ')}
          ${metadataJsonColumn ? `, "${metadataJsonColumn}"` : ''}
        )
        SELECT * FROM UNNEST (
          $1::uuid[],
          $2::text[],
          $3::vector[],
          ${metadataColumns.map((_, index) => `$${index + 4}::text[]`).join(', ')}
          ${metadataJsonColumn ? `, $${metadataColumns.length + 4}::jsonb[]` : ''}
        ) AS t (
          "${idColumn}",
          "${contentColumn}",
          "${embeddingColumn}",
          ${metadataColumns.map(col => `"${col}"`).join(', ')}
          ${metadataJsonColumn ? `, "${metadataJsonColumn}"` : ''}
        )
        `,
        [
          batchIds,
          batch.map(doc => doc.content),
          batch.map(doc => doc.embedding),
          ...metadataColumns.map(col => batch.map(doc => doc.metadata[col] || null)),
          ...(metadataJsonColumn ? [batch.map(doc => doc.metadata || {})] : []),
        ]
      );
    }
    return ids;
  }

  return defineAction(
    ai.registry,
    {
      name: `postgresIndexer/${tableName}`,
      actionType: 'indexer',
      inputSchema: z.array(z.any()),
      outputSchema: z.object({
        success: z.boolean(),
        count: z.number(),
      }),
      metadata: {
        description: 'Postgres vector indexer with pgvector support',
        config: params,
      },
    },
    async (documents) => {
      await addDocuments(documents);
      return { success: true, count: documents.length };
    }
  );
}

/**
 * Postgres plugin that provides a Postgres indexer
 */
export function postgres<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: PostgresIndexerConfig<EmbedderCustomOptions>[]
): GenkitPlugin {
  return genkitPlugin('postgres', async (ai: Genkit) => {
    for (const config of params) {
      configurePostgresIndexer(ai, {
        ...config,
        embedder: typeof config.embedder === 'string' 
          ? config.embedder 
          : async (content: string) => {
              if (typeof config.embedder === 'function') {
                const result = await (config.embedder as unknown as (input: { input: { content: { text: string }[] }[] }) => Promise<{ embedding: number[] }[]>)({ 
                  input: [{ content: [{ text: content }] }] 
                });
                return result[0]?.embedding || [];
              }
              return [];
            },
      });
    }
  });
}

export default postgres;