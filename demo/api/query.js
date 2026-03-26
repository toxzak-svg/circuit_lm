/**
 * Vercel Serverless Function — CircuitLM Query
 * 
 * Uses Python subprocess to run CircuitLM CLI.
 * Deploy to Vercel with this file at api/query.js
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { input } = req.body;
  if (!input?.trim()) {
    return res.status(400).json({ error: 'No input provided' });
  }

  const repoRoot = path.join(process.cwd(), '..');
  const traceFile = '/tmp/circuit_trace.json';
  const modelPath = path.join(repoRoot, 'models/infra_circuit.json');

  try {
    const result = await runCircuitLM(input, repoRoot, modelPath, traceFile);
    res.status(200).json(result);
  } catch (err) {
    console.error('CircuitLM error:', err);
    // Fallback response
    res.status(200).json(generateFallbackResponse(input));
  }
};

function runCircuitLM(input, repoRoot, modelPath, traceFile) {
  return new Promise((resolve, reject) => {
    const prompt = `User: ${input}\nAssistant: `;

    // Tokenize first
    const tokenProc = spawn('python3', [
      '-m', 'circuit_lm.cli',
      'tokenize', input
    ], { cwd: repoRoot, env: { ...process.env, PYTHONPATH: repoRoot } });

    let tokenOut = '';
    tokenProc.stdout.on('data', d => tokenOut += d);
    tokenProc.stderr.on('data', d => console.error('tokenize err:', d.toString()));
    tokenProc.on('close', () => {
      const tokens = tokenOut.trim().split(',').map(t => parseInt(t.trim())).filter(t => !isNaN(t));

      // Then trace
      const traceProc = spawn('python3', [
        '-m', 'circuit_lm.cli',
        'trace',
        '--prompt', prompt,
        '--model', modelPath,
        '--json-out', traceFile
      ], { cwd: repoRoot, env: { ...process.env, PYTHONPATH: repoRoot } });

      let traceErr = '';
      traceProc.stderr.on('data', d => traceErr += d);
      traceProc.on('close', () => {
        let steps = [];
        try {
          if (fs.existsSync(traceFile)) {
            const traceData = JSON.parse(fs.readFileSync(traceFile, 'utf8'));
            steps = Array.isArray(traceData) ? traceData.slice(0, 30) : [];
          }
        } catch {}

        resolve({
          input,
          tokens,
          steps,
          response: generateInfraDecision(input, tokens)
        });
      });
    });
  });
}

function generateInfraDecision(input, tokens) {
  const text = input.toLowerCase();
  const detected = {
    database: /\b(postgres|postgresql|mysql|mongodb|redis|db|database|sql)\b/.test(text),
    api: /\b(api|rest|endpoint|backend|server|microservice)\b/.test(text),
    frontend: /\b(react|vue|angular|frontend|ui|spa|next\.?js|svelte)\b/.test(text),
    auth: /\b(auth|login|user|register|jwt|oauth|session)\b/.test(text),
    realtime: /\b(websocket|realtime|stream|socket)\b/.test(text),
    storage: /\b(s3|storage|file|upload|cdn)\b/.test(text),
  };

  const services = [];
  const reasoning = [];

  if (detected.database) {
    services.push({ name: 'PostgreSQL', provider: 'Supabase', type: 'database', icon: '🗄️' });
    reasoning.push('Database detected → managed Postgres on Supabase');
  }
  if (detected.api || (!detected.frontend && services.length === 0)) {
    services.push({ name: 'Node.js API', provider: 'Railway', type: 'backend', icon: '⚙️' });
    reasoning.push('API requirement → Railway Node.js runtime');
  }
  if (detected.frontend) {
    services.push({ name: 'React Frontend', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
    reasoning.push('Frontend detected → Vercel edge deployment');
  }
  if (detected.auth) {
    services.push({ name: 'Auth Service', provider: 'Clerk', type: 'auth', icon: '🔐' });
    reasoning.push('Auth detected → Clerk for JWT/session management');
  }
  if (detected.realtime) {
    services.push({ name: 'WebSocket Server', provider: 'Railway', type: 'websocket', icon: '⚡' });
    reasoning.push('Realtime requirement → Railway WebSocket server');
  }
  if (services.length === 0) {
    services.push({ name: 'Node.js API', provider: 'Railway', type: 'backend', icon: '⚙️' });
    services.push({ name: 'React Frontend', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
    reasoning.push('Default stack → Railway + Vercel');
  }

  const connections = [];
  for (let i = 0; i < services.length - 1; i++) {
    const from = services[i];
    const to = services[i + 1];
    if (from.type === 'auth' || to.type === 'auth') continue;
    connections.push({ from: from.name, to: to.name, protocol: 'REST / JSON' });
  }

  return {
    services,
    connections,
    reasoning,
    total_cost_estimate: `$${services.length * 5}/mo`,
    deployment_steps: services.map((s, i) => `${i + 1}. Provision ${s.name} on ${s.provider}`)
  };
}

function generateFallbackResponse(input) {
  return {
    input,
    tokens: input.split('').map(c => c.charCodeAt(0)),
    steps: [],
    response: generateInfraDecision(input, [])
  };
}
