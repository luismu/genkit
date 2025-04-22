import { PostgresIndexer } from '../indexer.js';
import { PostgresEngine, PostgresEngineArgs } from '../engine.js';
import { Document } from '@genkit-ai/ai/retriever';
import { EmbedderFn } from '@genkit-ai/ai/embedder';
import * as dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const TEST_TABLE = "test_embeddings";

const embeddingSchema = z.object({
  embedding: z.array(z.number())
});

const pgArgs: PostgresEngineArgs = {
  user: process.env.DB_USER ?? "",
  password: process.env.PASSWORD ?? ""
};

describe("PostgresIndexer Integration Tests", () => {
  let indexer: PostgresIndexer;
  let PEInstance: PostgresEngine;

  // Mock embedder function for testing
  const mockEmbedder: EmbedderFn<typeof embeddingSchema> = async (docs: Document[]) => {
    return {
      embeddings: [{
        embedding: [0.1, 0.2, 0.3, 0.4, 0.5]
      }]
    };
  };

  beforeAll(async () => {
    // Create table for testing
    PEInstance = await PostgresEngine.fromInstance(
      process.env.PROJECT_ID ?? "",
      process.env.REGION ?? "",
      process.env.INSTANCE_NAME ?? "",
      process.env.DB_NAME ?? "",
      pgArgs
    );

    await PEInstance.pool.raw(`DROP TABLE IF EXISTS "${TEST_TABLE}"`);
    await PEInstance.pool.raw(`
      CREATE TABLE IF NOT EXISTS "${TEST_TABLE}" (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding REAL[],
        metadata JSONB
      );
    `);

    // Initialize indexer
    indexer = new PostgresIndexer({
      tableName: TEST_TABLE,
      embedder: mockEmbedder,
      connectionParams: {
        projectId: process.env.PROJECT_ID ?? "",
        region: process.env.REGION ?? "",
        instanceName: process.env.INSTANCE_NAME ?? "",
        database: process.env.DB_NAME ?? "",
        args: pgArgs
      }
    });

    await indexer.initialize();
  });

  test('should insert documents with embeddings', async () => {
    const documents = [
      new Document({ 
        content: [{ text: 'Test document 1' }],
        metadata: { source: 'test1' }
      }),
      new Document({ 
        content: [{ text: 'Test document 2' }],
        metadata: { source: 'test2' }
      })
    ];

    await indexer.index(documents);

    // Verify documents were inserted
    const result = await PEInstance.pool
      .select('*')
      .from(TEST_TABLE)
      .orderBy('id');

    expect(result).toHaveLength(2);
    expect(result[0]).toMatchObject({
      content: 'Test document 1',
      metadata: { source: 'test1' }
    });
    expect(result[0].embedding).toHaveLength(5);
    expect(result[1]).toMatchObject({
      content: 'Test document 2',
      metadata: { source: 'test2' }
    });
  });

  test('should handle batch insertions', async () => {
    const documents = Array.from({ length: 5 }, (_, i) => 
      new Document({ 
        content: [{ text: `Batch document ${i + 1}` }],
        metadata: { batch: i + 1 }
      })
    );

    await indexer.index(documents, { batchSize: 2 });

    const result = await PEInstance.pool
      .select('*')
      .from(TEST_TABLE)
      .whereRaw(`metadata->>'batch' IS NOT NULL`)
      .orderBy('id');

    expect(result).toHaveLength(5);
    expect(result.map(r => r.metadata.batch)).toEqual([1, 2, 3, 4, 5]);
  });

  afterAll(async () => {
    await indexer.close();
    await PEInstance.pool.raw(`DROP TABLE IF EXISTS "${TEST_TABLE}"`);
    await PEInstance.closeConnection();
  });
});