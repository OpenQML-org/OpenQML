CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY,
  name TEXT,
  category TEXT,
  tagline TEXT,
  description TEXT,
  wins TEXT,
  "limits" TEXT,
  framework TEXT,
  contact TEXT,
  type TEXT,
  status TEXT,
  createdAt TEXT
);

CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY,
  title TEXT,
  author TEXT,
  userId TEXT,
  body TEXT,
  createdAt TEXT
);

CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY,
  threadId TEXT,
  author TEXT,
  userId TEXT,
  body TEXT,
  createdAt TEXT
);

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  provider TEXT,
  providerId TEXT,
  name TEXT,
  email TEXT,
  avatar TEXT,
  createdAt TEXT
);

CREATE TABLE IF NOT EXISTS votes (
  userId TEXT,
  itemId TEXT,
  PRIMARY KEY (userId, itemId)
);
