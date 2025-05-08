# Cloud SQL for PostgreSQL plugin

The Cloud SQL for PostgreSQL plugin provides indexer and retriever implementations that use PostgreSQL with the pgvector extension for vector similarity search.

## Installation

```posix-terminal
npm i --save @genkitx-cloud-sql-pg
```

## Configuration

To use this plugin, first create a PostgresEngine instance:

```ts
import { PostgresEngine } from '@genkit-ai/cloud-sql-pg';

// Create PostgresEngine instance
const engine = await PostgresEngine.fromEngineArgs({
  user: 'postgres',
  password: 'password',
  host: 'localhost',
  database: 'mydb',
  port: 5432
});
```

Then, specify the plugin when you initialize Genkit:

```ts
import { genkit } from 'genkit';
import { postgres } from '@genkit-ai/cloud-sql-pg';
import { textEmbedding004 } from '@genkit-ai/vertexai';

const ai = genkit({
  plugins: [
    postgres([
      {
        tableName: 'my-documents',
        engine: engine, // Use the PostgresEngine instance
        embedder: textEmbedding004,
        schemaName: 'public', 
        contentColumn: 'content',
        embeddingColumn: 'embedding',
        metadataColumns: ['source', 'category'],
        ignoreMetadataColumns: ['created_at', 'updated_at'],
        idColumn: 'id',
        metadataJsonColumn: 'metadata',
        distanceStrategy: 'cosine',
      },
    ]),
  ],
});
```

The plugin validates the following when initializing:
- Required columns exist (id, content, embedding)
- Content column is a text type
- Embedding column is a vector type
- Metadata columns exist (if specified)
- Cannot use both metadataColumns and ignoreMetadataColumns
- If ignoreMetadataColumns is used, it will use all remaining columns as metadata

## Usage

Import retriever and indexer references like so:

```ts
import { postgresRetrieverRef, postgresIndexerRef } from '@genkit-ai/cloud-sql-pg';
```

Then, use these references with `ai.retrieve()` and `ai.index()`:

```ts
// To use the table you configured when you loaded the plugin:
let docs = await ai.retrieve({ retriever: postgresRetrieverRef, query });

// To specify a table:
export const myDocumentsRetriever = postgresRetrieverRef({
  tableName: 'my-documents',
});
docs = await ai.retrieve({ retriever: myDocumentsRetriever, query });
```

```ts
// To use the table you configured when you loaded the plugin:
await ai.index({ indexer: postgresIndexerRef, documents });

// To specify a table:
export const myDocumentsIndexer = postgresIndexerRef({
  tableName: 'my-documents',
});
await ai.index({ indexer: myDocumentsIndexer, documents });

// Index with custom batch size
await ai.index({ 
  indexer: myDocumentsIndexer, 
  documents,
  options: { batchSize: 10 }
});

// Index with custom ID from metadata
const docWithCustomId = new Document({
  content: [{ text: 'Document with custom ID' }],
  metadata: { 
    source: 'test',
    customId: 'custom-123' // This will be used as the document ID
  }
});
await ai.index({ 
  indexer: myDocumentsIndexer, 
  documents: [docWithCustomId]
});
```

See the [Retrieval-augmented generation](../rag.md) page for a general discussion on indexers and retrievers. 