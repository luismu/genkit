import { Genkit, z } from 'genkit';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import { EmbedderArgument, Embedding } from 'genkit/embedder';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from 'genkit/retriever';

import { v4 as uuidv4 } from 'uuid';
import { PostgresEngine } from './engine';

const PostgresRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
  filter: z.record(z.string(), z.any()).optional(),
});

const PostgresIndexerOptionsSchema = z.object({
});

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

/**
 * Postgres plugin that provides a Postgres retriever and indexer
 * @param params An array of params to set up Postgres retrievers and indexers
 * @param params.tableName The name of the table
 * @param params.embedder The embedder to use for the indexer and retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns The Postgres Genkit plugin
 */
export function postgres<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: {
    tableName: string,
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    engine: PostgresEngine;
    schemaName?: string;
    contentColumn: string;
    embeddingColumn: string;
    metadataColumns?: string[];
    idColumn: string;
    metadataJsonColumn?: string;
    distanceStrategy: 'cosine' | 'ip' | 'l2';
  }[]
): GenkitPlugin {
  return genkitPlugin('postgres', async (ai: Genkit) => {
    params.map((i) => configurePostgresRetriever(ai, i));
    params.map((i) => configurePostgresIndexer(ai, i));
  });
}

export default postgres;

/**
 * Configures a Postgres retriever.
 * @param ai A Genkit instance
 * @param params The params for the retriever
 * @param params.tableName The name of the table
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns A Postgres retriever
 */
export function configurePostgresRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {

  return ai.defineRetriever(
    {
      name: `postgres/${params.tableName}`,
      configSchema: PostgresRetrieverOptionsSchema,
    },
    async (content, options) => {
      // Add logic for handling content and options here
      console.log(`Retrieving data for table: ${params.tableName}`);
      return {
        documents: [], // Return appropriate documents based on your logic
      };
    }
  );
}

/**
 * Configures a Postgres indexer.
 * @param ai A Genkit instance
 * @param params The params for the indexer
 * @param params.tableName The name of the indexer
 * @param params.engine The engine to use for the indexer
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @param params.metadataColumns The metadata columns to use for the indexer
 * @param params.idColumn The id column to use for the indexer
 * @param params.metadataJsonColumn The metadata json column to use for the indexer
 * @param params.distanceStrategy The distance strategy to use for the indexer
 * @param params.contentColumn The content column to use for the indexer
 * @param params.embeddingColumn The embedding column to use for the indexer
 * @param params.schemaName The schema name to use for the indexer
 * @param params.chunkSize The chunk size to use for the indexer
 * @returns Add documents to vector store
 * 
 * 
 * 
 */
export function configurePostgresIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    tableName: string;
    engine: PostgresEngine;
    schemaName?: string;
    contentColumn: string;
    embeddingColumn: string;
    metadataColumns?: string[];
    idColumn: string;
    metadataJsonColumn?: string;
    distanceStrategy: 'cosine' | 'ip' | 'l2';
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const {
    tableName,
    engine,
    schemaName = 'public',
    contentColumn,
    embeddingColumn,
    metadataColumns = [],
    idColumn,
    metadataJsonColumn,
    distanceStrategy,
    embedder,
    embedderOptions
  } = params;

  async function ensureTableExists() {
    await engine.pool.raw('CREATE EXTENSION IF NOT EXISTS vector');
    
    await engine.pool.raw(`
      CREATE TABLE IF NOT EXISTS ${schemaName}.${tableName} (
        ${idColumn} UUID PRIMARY KEY,
        ${contentColumn} TEXT NOT NULL,
        ${embeddingColumn} VECTOR(1536),
        ${metadataJsonColumn ? `${metadataJsonColumn} JSONB,` : ''}
        ${metadataColumns.map(col => `${col} TEXT,`).join('\n')}
        created_at TIMESTAMP DEFAULT NOW()
      );
      
      CREATE INDEX IF NOT EXISTS idx_${tableName}_embedding 
      ON ${schemaName}.${tableName} 
      USING ivfflat (${embeddingColumn}) 
      WITH (lists = 100);
    `);
  }

  return ai.defineIndexer(
    {
      name: `postgres/${params.tableName}`,
      configSchema: z.object({}).optional(),
    },
    async (docs, options) => {
      await ensureTableExists();

      // Generate embeddings for all documents
      const embeddings = await Promise.all(
        docs.map(doc => 
          ai.embed({
            embedder,
            content: { content: doc.content },
            options: embedderOptions
          })
        )
      );

      // Prepare values for batch insert
      const values = docs.map((doc, i) => {
        const docEmbeddings = embeddings[i];
        const metadata = doc.metadata || {};
        
        return {
          id: metadata[idColumn] || uuidv4(),
          content: doc.content,
          embedding: JSON.stringify(docEmbeddings[0].embedding),
          metadata,
          ...metadata
        };
      });

      const columns = [
        idColumn,
        contentColumn,
        embeddingColumn,
        ...(metadataJsonColumn ? [metadataJsonColumn] : []),
        ...metadataColumns
      ];

      const insertQuery = `
        INSERT INTO ${schemaName}.${tableName} (${columns.join(', ')})
        VALUES ${values.map((_, i) => 
          `(${columns.map((_, j) => `$${i * columns.length + j + 1}`).join(', ')})`
        ).join(', ')}
        ON CONFLICT (${idColumn}) DO UPDATE SET
          ${contentColumn} = EXCLUDED.${contentColumn},
          ${embeddingColumn} = EXCLUDED.${embeddingColumn}
          ${metadataJsonColumn ? `, ${metadataJsonColumn} = EXCLUDED.${metadataJsonColumn}` : ''}
      `;

      const flatValues = values.flatMap(value => [
        value.id,
        value.content,
        value.embedding,
        ...(metadataJsonColumn ? [JSON.stringify(value.metadata)] : []),
        ...metadataColumns.map(col => value[col] || null)
      ]);

      await engine.pool.raw(insertQuery, flatValues);
    }
  );
}
