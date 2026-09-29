Vectors

A vector is an ordered collection of numbers. In machine learning and embeddings, vectors are used to represent data numerically.

For example, a simple vector can be represented as:

[2, 5, 8]

This vector has three dimensions because it contains three numbers.

In machine learning, vectors can represent features of an object. For example, a student could be represented using a vector containing their study hours, attendance, and test score:

[5, 90, 85]

Here, 5 could represent study hours, 90 could represent attendance percentage, and 85 could represent a test score.

Vectors are especially important in text embeddings. An embedding model converts text into a vector of numbers. For example, the sentence "I love learning Python" could be converted into a vector such as:

[0.21, -0.43, 0.87, 0.15, -0.62]

Real embedding vectors usually contain hundreds or thousands of dimensions.

The numbers in an embedding vector do not normally have individual human-readable meanings. Instead, the entire vector represents patterns and semantic information about the original text.

Similar texts tend to have similar vectors. For example:

"I want to learn Python."

"I am interested in studying Python."

These sentences have similar meanings, so their embedding vectors should be relatively close in vector space.

Vector similarity is used to determine how closely two pieces of data are related. Common similarity methods include cosine similarity, Euclidean distance, and dot product.

Cosine similarity compares the direction of two vectors. If two vectors point in similar directions, their cosine similarity is high.

Euclidean distance measures the physical distance between two points in vector space. A smaller distance generally means the vectors are more similar.

In a RAG system, document chunks are converted into vectors using an embedding model. These vectors are stored in a vector database such as ChromaDB.

When a user asks a question, the question is also converted into a vector. The system compares the query vector with the stored document vectors and retrieves the most similar chunks.

For example, if a document contains information about Python programming and the user asks:

"How can I learn Python?"

The query is converted into a vector. The vector database searches for document vectors that are semantically similar to the query.

The retrieved text is then given to the language model as context so that it can generate an answer.

Therefore, vectors are a fundamental part of modern machine learning, embeddings, semantic search, vector databases, and Retrieval-Augmented Generation systems.