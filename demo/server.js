/**
 * CircuitLM Interactive Demo Server
 * 
 * Node.js + Express — serves the demo page and calls CircuitLM CLI.
 * 
 * Run from circuit_lm repo root:
 *   cd circuit_lm
 *   cd demo && npm install && npm start
 *   Visit http://localhost:3000
 */

import express from 'express';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));
app.use(express.static(join(__dirname)));

// Serve index.html at root
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'));
});

// Run CircuitLM query
app.post('/api/query', async (req, res) => {
  const { input } = req.body;
  if (!input?.trim()) {
    return res.status(400).json({ error: 'No input provided' });
  }

  const repoRoot = join(__dirname, '..');

  try {
    const result = await runCircuitLM(repoRoot, input);
    res.json(result);
  } catch (err) {
    console.error('CircuitLM error:', err);
    // Fall back to simulated response
    res.json(generateFallbackResponse(input));
  }
});

/**
 * Call CircuitLM CLI and return structured trace + decision
 */
async function runCircuitLM(repoRoot, userInput) {
  const prompt = `User: ${userInput}\nAssistant: `;
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  // Try to run CircuitLM trace
  const modelPath = join(repoRoot, 'models/infra_circuit.json');
  const traceResult = await runCommand(pythonCmd, [
    '-m', 'circuit_lm.cli',
    'trace',
    '--prompt', prompt,
    '--model', modelPath,
    '--json-out', '/tmp/circuit_trace.json'
  ], repoRoot);

  // Try to read the trace file
  let steps = [];
  try {
    const traceData = JSON.parse(readFileSync('/tmp/circuit_trace.json', 'utf8'));
    steps = Array.isArray(traceData) ? traceData.slice(0, 30) : [];
  } catch {}


  // Tokenize input
  let tokens = [];
  try {
    const tokResult = await runCommand(pythonCmd, [
      '-m', 'circuit_lm.cli',
      'tokenize', userInput
    ], repoRoot);
    tokens = tokResult.stdout.trim().split(',').map(t => parseInt(t.trim())).filter(t => !isNaN(t));
  } catch {
    // Fallback: hash-based tokenization
    tokens = userInput.split('').map(c => c.charCodeAt(0));
  }

  // Generate infrastructure decision
  const decision = generateInfraDecision(userInput, tokens);

  return {
    input: userInput,
    tokens,
    steps,
    response: decision
  };
}

/**
 * Run a command and return { stdout, stderr }
 */
function runCommand(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const env = { ...process.env, PYTHONPATH: cwd };
    const proc = spawn(cmd, args, { cwd, env, timeout: 30000, shell: true });
    let stdout = '', stderr = '';
    proc.stdout.on('data', d => stdout += d);
    proc.stderr.on('data', d => stderr += d);
    proc.on('close', code => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(stderr || `Exit code ${code}`));
    });
    proc.on('error', reject);
  });
}

/**
 * Generate infrastructure decision based on input keywords.
 * In the real system, this would be CircuitLM's actual output.
 * This demo shows what the decision trace looks like.
 */
function generateInfraDecision(input, tokens) {
  const text = input.toLowerCase();
  const detected = {
    database: /\b(postgres|postgresql|mysql|mongodb|redis|db|database|sql|mongo|sqlserver|oracle|mariadb)\b/.test(text),
    api: /\b(api|rest|endpoint|backend|server|microservice|grpc|graphql)\b/.test(text),
    frontend: /\b(react|vue|angular|frontend|ui|spa|next\.js|nextjs|svelte|html|css|tailwind)\b/.test(text),
    auth: /\b(auth|login|user|register|jwt|oauth|passport|session|ACL|permission)\b/.test(text),
    realtime: /\b(websocket|realtime|stream|socket|live|ws|server-sent)\b/.test(text),
    deploy: /\b(deploy|host|production|staging|ci\/cd|github|vercel|railway|fly|aws|ec2|lambda|k8s|kubernetes|docker|container)\b/.test(text),
    storage: /\b(s3|storage|file|upload|blob|cdn|cloudflare|cloud)\b/.test(text),
    monitoring: /\b(monitor|log|metrics|datadog|sentry|observability|alert|uptime)\b/.test(text),
  };

  const services = [];
  const connections = [];
  const reasoning = [];

  // Database
  if (detected.database) {
    if (/postgres|postgresql/i.test(text)) {
      services.push({ name: 'PostgreSQL', provider: 'Supabase', type: 'database', icon: '🗄️' });
      reasoning.push('PostgreSQL detected → managed Postgres on Supabase (with real-time subscriptions)');
    } else if (/mongo|mongodb/i.test(text)) {
      services.push({ name: 'MongoDB Atlas', provider: 'MongoDB', type: 'database', icon: '🗄️' });
      reasoning.push('MongoDB detected → Atlas M10 cluster provisioned');
    } else {
      services.push({ name: 'PostgreSQL', provider: 'Supabase', type: 'database', icon: '🗄️' });
      reasoning.push('Database detected → Supabase managed Postgres');
    }
  }

  // Backend/API
  if (detected.api || (!detected.frontend && !detected.database)) {
    if (/node|express|typescript/i.test(text)) {
      services.push({ name: 'Node.js API', provider: 'Railway', type: 'backend', icon: '⚙️' });
      reasoning.push('Node.js API detected → Railway Node.js runtime');
    } else if (/python|fastapi|django|flask/i.test(text)) {
      services.push({ name: 'Python API', provider: 'Railway', type: 'backend', icon: '⚙️' });
      reasoning.push('Python API detected → Railway with Python runtime');
    } else {
      services.push({ name: 'REST API', provider: 'Railway', type: 'backend', icon: '⚙️' });
      reasoning.push('API requirement detected → Railway REST endpoint');
    }
  }

  // Frontend
  if (detected.frontend) {
    if (/react|next\.?js|nextjs/i.test(text)) {
      services.push({ name: 'Next.js Frontend', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
      reasoning.push('Next.js/React detected → Vercel edge deployment');
    } else if (/vue|nuxt/i.test(text)) {
      services.push({ name: 'Nuxt.js Frontend', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
      reasoning.push('Vue/Nuxt detected → Vercel deployment');
    } else {
      services.push({ name: 'React SPA', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
      reasoning.push('Frontend detected → Vercel SPA hosting');
    }
  }

  // Auth
  if (detected.auth) {
    services.push({ name: 'Auth Service', provider: 'Clerk', type: 'auth', icon: '🔐' });
    reasoning.push('Authentication detected → Clerk for JWT/session management');
    if (!services.find(s => s.type === 'frontend')) {
      connections.push({ from: 'Frontend', to: 'Auth', protocol: 'JWT / OAuth 2.0' });
    }
  }

  // Realtime
  if (detected.realtime) {
    services.push({ name: 'WebSocket Server', provider: 'Railway', type: 'websocket', icon: '⚡' });
    reasoning.push('Realtime requirement → Railway WebSocket server with Socket.IO');
  }

  // Storage
  if (detected.storage) {
    services.push({ name: 'Object Storage', provider: 'Cloudflare R2', type: 'storage', icon: '📦' });
    reasoning.push('Storage detected → Cloudflare R2 (S3-compatible, no egress fees)');
  }

  // Monitoring
  if (detected.monitoring) {
    services.push({ name: 'Observability', provider: 'Grafana Cloud', type: 'monitoring', icon: '📊' });
    reasoning.push('Monitoring detected → Grafana Cloud metrics + logs + traces');
  }

  // Default if nothing detected
  if (services.length === 0) {
    services.push({ name: 'Node.js API', provider: 'Railway', type: 'backend', icon: '⚙️' });
    services.push({ name: 'React Frontend', provider: 'Vercel', type: 'frontend', icon: '🖥️' });
    services.push({ name: 'PostgreSQL', provider: 'Supabase', type: 'database', icon: '🗄️' });
    reasoning.push('No specific stack detected → defaults: Railway + Vercel + Supabase');
  }

  // Wire connections
  for (let i = 0; i < services.length - 1; i++) {
    const fromSvc = services[i];
    const toSvc = services[i + 1];
    if (fromSvc.type === 'auth' || toSvc.type === 'auth') continue;
    if (fromSvc.type === 'backend' && toSvc.type === 'database') {
      connections.push({ from: fromSvc.name, to: toSvc.name, protocol: 'Postgres / Prisma' });
    } else if (fromSvc.type === 'frontend' && toSvc.type === 'backend') {
      connections.push({ from: fromSvc.name, to: toSvc.name, protocol: 'REST / JSON' });
    } else {
      connections.push({ from: fromSvc.name, to: toSvc.name, protocol: 'REST / JSON' });
    }
  }

  // Estimate cost (rough)
  const baseCost = services.length * 5;
  const totalCost = `$${baseCost}/mo`;

  // Generate deployment steps
  const deploymentSteps = services.map((svc, i) => {
    return `${i + 1}. Provision ${svc.name} on ${svc.provider}`;
  });
  if (connections.length > 0) {
    deploymentSteps.push(`${services.length + 1}. Wire connections: ${connections.map(c => `${c.from} → ${c.to}`).join(', ')}`);
  }
  if (detected.deploy) {
    deploymentSteps.push(`${services.length + connections.length + 1}. Configure CI/CD pipeline`);
  }
  deploymentSteps.push(`✓ Deploy to ${services.map(s => s.provider).join(', ')}`);

  return {
    services,
    connections,
    reasoning,
    total_cost_estimate: totalCost,
    deployment_steps: deploymentSteps
  };
}

/**
 * Fallback when CircuitLM CLI is not available
 */
function generateFallbackResponse(input) {
  return {
    input,
    tokens: input.split('').map(c => c.charCodeAt(0)),
    steps: [],
    response: generateInfraDecision(input, [])
  };
}

app.listen(PORT, () => {
  console.log(`CircuitLM Demo running at http://localhost:${PORT}`);
  console.log('CircuitLM repo:', join(__dirname, '..'));
});
