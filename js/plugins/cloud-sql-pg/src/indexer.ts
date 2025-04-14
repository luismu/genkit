import { Document } from '@genkit-ai/ai/retriever';
import { EmbedderFn } from '@genkit-ai/ai/embedder';
import PostgresEngine, { PostgresEngineArgs } from './engine.js';
import { z } from 'zod';

const embeddingSchema = z.object({
  embedding: z.array(z.number())
});

export interface GCloudConnectionParams {
  projectId: string;
  region: string;
  instanceName: string;
  database: string;
  args?: PostgresEngineArgs;
}

export interface IndexedDocument {
  content: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export class GCloudIndexer {
  private engine!: PostgresEngine;
  private embedder: EmbedderFn<typeof embeddingSchema>;

  constructor(
    private config: {
      tableName: string;
      embedder: EmbedderFn<typeof embeddingSchema>;
      connectionParams: GCloudConnectionParams;
    }
  ) {
    this.embedder = config.embedder;
  }

  async initialize() {
    this.engine = await PostgresEngine.fromInstance(
      this.config.connectionParams.projectId,
      this.config.connectionParams.region,
      this.config.connectionParams.instanceName,
      this.config.connectionParams.database,
      this.config.connectionParams.args || {}
    );
  }

  private async generateEmbeddings(documents: Document[]): Promise<IndexedDocument[]> {
    const embeddings = await Promise.all(
      documents.map(async (doc) => {
        const content = doc.text;
        const result = await this.embedder([doc]);
        return {
          content,
          embedding: result.embeddings[0].embedding,
          metadata: doc.metadata
        };
      })
    );
    return embeddings;
  }

  async index(documents: Document[], options?: { batchSize?: number }) {
    if (!this.engine) {
      throw new Error('Indexer not initialized. Call initialize() first.');
    }

    const vectors = await this.generateEmbeddings(documents);
    const batchSize = options?.batchSize || 100;

    // Process documents in batches
    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize);
      
      await this.engine.pool.table(this.config.tableName).insert(
        batch.map(doc => ({
          content: doc.content,
          embedding: doc.embedding,
          metadata: doc.metadata
        }))
      );
    }
  }

  async close() {
    if (this.engine) {
      await this.engine.closeConnection();
    }
  }
}