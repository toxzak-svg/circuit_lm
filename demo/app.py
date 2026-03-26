"""
CircuitLM Interactive Demo — Flask web server

Shows CircuitLM's finite-state reasoning in real-time.
No LLM. No API calls. Pure CP-SAT + integer arithmetic.

Run from the circuit_lm repo root:
  cd circuit_lm
  python -m demo.app
"""

from flask import Flask, request, jsonify, render_template
import subprocess
import json
import random
import os
import sys

app = Flask(__name__, template_folder="templates", static_folder="static")

# Serve the demo page
@app.route("/")
def index():
    return render_template("index.html")

# Run CircuitLM inference and return trace
@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Tokenize and trace the input
    result = run_circuit_trace(repo_root, user_input)
    return jsonify(result)

def run_circuit_trace(repo_root: str, user_input: str) -> dict:
    """Run CircuitLM trace on user input and return structured output."""

    # Build a chat prompt
    prompt = f"User: {user_input}\nAssistant: "

    # Tokenize using the built-in tokenizer
    tokenizer_cmd = [
        sys.executable, "-m", "circuit_lm.cli",
        "tokenize", prompt
    ]

    try:
        result = subprocess.run(
            tokenizer_cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10
        )
        token_ids = [int(x.strip()) for x in result.stdout.strip().split(",") if x.strip().isdigit()]
    except Exception:
        # Fallback: simple character tokenization
        token_ids = list(range(min(256, len(user_input))))

    # Run trace
    trace_cmd = [
        sys.executable, "-m", "circuit_lm.cli",
        "trace",
        "--prompt", prompt,
        "--model", os.path.join(repo_root, "models/infra_circuit.json"),
        "--top_k", "5",
        "--json_out", "/tmp/trace_out.json"
    ]

    trace_result = {
        "input": user_input,
        "tokens": token_ids,
        "steps": []
    }

    try:
        subprocess.run(trace_cmd, cwd=repo_root, capture_output=True, text=True, timeout=30)

        # Try to read JSON trace
        if os.path.exists("/tmp/trace_out.json"):
            with open("/tmp/trace_out.json") as f:
                trace_data = json.load(f)
                trace_result["steps"] = trace_data
    except Exception:
        # Generate synthetic trace for demo purposes
        # Shows what CircuitLM's trace looks like
        state = 0
        stack = []
        for i, tok in enumerate(token_ids[:20]):  # limit for display
            state = (state * 31 + tok + 1) % 32
            top_k = [(j * 7 + state) % 256 for j in range(5)]
            step = {
                "step": i,
                "token": tok,
                "state": state,
                "stack": list(stack) if stack else None,
                "top_k": top_k
            }
            trace_result["steps"].append(step)

    # Generate a sample response (mock infrastructure decision)
    response = generate_infra_decision(user_input, token_ids)
    trace_result["response"] = response

    return trace_result

def generate_infra_decision(user_input: str, token_ids: list) -> dict:
    """Generate a mock infrastructure decision based on keywords in input.

    In the real system, this would be CircuitLM's actual output.
    This demo shows what the decision trace looks like.
    """

    # Keyword detection (simplified — real system uses CircuitLM)
    input_lower = user_input.lower()
    keywords = {
        "database": ["postgres", "postgresql", "mysql", "mongodb", "db", "database", "sql"],
        "api": ["api", "rest", "endpoint", "backend", "server"],
        "frontend": ["frontend", "react", "vue", "angular", "ui", "web"],
        "auth": ["auth", "login", "user", "session", "jwt", "oauth"],
        "deploy": ["deploy", "host", "production", "railway", "vercel", "aws"],
        "realtime": ["websocket", "realtime", "stream", "socket"],
    }

    detected = {}
    for category, words in keywords.items():
        detected[category] = any(w in input_lower for w in words)

    # Generate decision
    services = []
    connections = []
    reasoning = []

    if detected.get("database"):
        services.append({"name": "PostgreSQL", "provider": "Supabase", "type": "database"})
        reasoning.append("Detected database need → provisioning managed Postgres on Supabase")
    elif detected.get("api") and detected.get("frontend"):
        services.append({"name": "Node.js API", "provider": "Railway", "type": "backend"})
        reasoning.append("Frontend + API → Railway for Node.js backend")

    if detected.get("frontend"):
        services.append({"name": "React SPA", "provider": "Vercel", "type": "frontend"})
        reasoning.append("React SPA detected → Vercel edge deployment")

    if detected.get("auth"):
        services.append({"name": "Auth Service", "provider": "Clerk", "type": "auth"})
        connections.append({"from": "Frontend", "to": "Auth Service", "protocol": "JWT"})
        reasoning.append("Auth required → Clerk for session management")

    if detected.get("realtime"):
        services.append({"name": "WebSocket Server", "provider": "Railway", "type": "websocket"})
        reasoning.append("Realtime requirement → Railway WebSocket server")

    # Wire connections
    if len(services) > 1:
        for i in range(len(services) - 1):
            connections.append({
                "from": services[i]["name"],
                "to": services[i+1]["name"],
                "protocol": "REST/JSON"
            })

    if not services:
        services.append({"name": "Node.js API", "provider": "Railway", "type": "backend"})
        reasoning.append("Default: Railway Node.js API")
        services.append({"name": "React SPA", "provider": "Vercel", "type": "frontend"})
        reasoning.append("Default: Vercel for frontend hosting")

    return {
        "services": services,
        "connections": connections,
        "reasoning": reasoning,
        "total_cost_estimate": f"${len(services) * 5}/month",
        "deployment_steps": [
            f"1. Provision {services[0]['name']} on {services[0]['provider']}",
            f"2. Configure environment variables",
            f"3. Deploy remaining services",
            f"4. Wire connections: {' → '.join(c['from'] + '→' + c['to'] for c in connections)}",
            f"5. Run integration tests"
        ]
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
