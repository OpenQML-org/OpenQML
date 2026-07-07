# Deploying OpenQML.org to Cloudflare Workers

One-time setup (from the project root):

```bash
npm install
npx wrangler login

# 1. Create the D1 database, copy the printed database_id
npx wrangler d1 create openqml
#    -> paste the id into wrangler.jsonc ("database_id")

# 2. Apply the schema to the remote database
npx wrangler d1 execute openqml --remote --file=./schema.sql

# 3. Set the admin token for /review (pick a strong secret)
npx wrangler secret put ADMIN_TOKEN
```

Deploy:

```bash
npm run deploy
```

Bind the domain: Cloudflare dashboard → Workers & Pages → openqml →
Settings → Domains & Routes → add `openqml.org` (the zone is already
on your account, so this is one click).

Local development (no Cloudflare needed — submissions fall back to
`data/submissions.json`):

```bash
echo 'ADMIN_TOKEN=choose-a-local-token' > .env.local   # enables /review locally
npm run dev
```

Note: if ADMIN_TOKEN is not set, the review API rejects all requests.

Local preview inside the Workers runtime (uses a local D1):

```bash
npx wrangler d1 execute openqml --local --file=./schema.sql
npm run preview
```


## Sign-in (OAuth)

The whole site requires login. Configure at least one provider; buttons appear
automatically for whichever providers have credentials set.

Set a session secret first (any long random string):

```bash
npx wrangler secret put SESSION_SECRET
```

### GitHub
1. github.com → Settings → Developer settings → OAuth Apps → New OAuth App
2. Callback URL: `https://openqml.org/api/auth/github/callback`
3. `npx wrangler secret put GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`

### Google
1. console.cloud.google.com → APIs & Services → Credentials → OAuth client ID (Web)
2. Authorized redirect URI: `https://openqml.org/api/auth/google/callback`
3. `npx wrangler secret put GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`

### Apple
1. developer.apple.com → Identifiers: create a Services ID (this is APPLE_CLIENT_ID),
   enable Sign in with Apple, set return URL `https://openqml.org/api/auth/apple/callback`
2. Keys: create a Sign in with Apple key, download the .p8 file
3. Secrets: `APPLE_CLIENT_ID` (Services ID), `APPLE_TEAM_ID`, `APPLE_KEY_ID`,
   and `APPLE_PRIVATE_KEY` (contents of the .p8; replace newlines with \n)

Apple requires a paid developer account and is the most involved of the three —
ship with GitHub + Google first if needed.

### Local development
`.env.local`:
```
SESSION_SECRET=any-long-string
ADMIN_TOKEN=choose-a-local-token
```
Dev login (name-only, no OAuth) is enabled automatically outside production,
or in production by setting `DEV_LOGIN=1` (do not do this on the real site).

### Opening parts of the site later
The login wall is `middleware.ts`. To make the catalog readable without an
account, add e.g. `/^\/$/, /^\/models/, /^\/algorithms/` to the PUBLIC list.
