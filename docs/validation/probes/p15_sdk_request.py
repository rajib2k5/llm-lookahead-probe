"""C15/C16: real anthropic SDK against a 127.0.0.1 stub; capture request body; parse response via ClaudeModel."""
import sys, os, json, threading, tempfile, inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
captured = []
RESP = {"id": "msg_stub", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "LONG"}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 123, "output_tokens": 2}}
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); captured.append((self.path, json.loads(self.rfile.read(n))))
        b = json.dumps(RESP).encode(); self.send_response(200)
        self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}"
import anthropic
from llm_lookahead_probe.live import MODEL_TABLE, ClaudeModel, ResponseCache, default_client_factory
fails = []
print("anthropic", anthropic.__version__)
sig = inspect.signature(anthropic.Anthropic().messages.create) if False else inspect.signature(anthropic.resources.messages.Messages.create)
print("create has thinking:", "thinking" in sig.parameters, "| has temperature:", "temperature" in sig.parameters,
      "| VAR_KEYWORD:", any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()))
info = MODEL_TABLE["claude-haiku-5-5"]
m = ClaudeModel(info, ResponseCache(tempfile.mkdtemp()), lambda: default_client_factory("test-key-not-real"), max_calls=5)
try:
    txt = m.complete("prompt text")
except Exception as e:
    fails.append(("complete raised", repr(e))); txt = None
print("returned text:", txt, "| usage:", m.usage()["input_tokens"], m.usage()["output_tokens"], dict(m.stop_reasons))
if captured:
    path, body = captured[-1]; print("POST", path, "body:", json.dumps(body))
    if path != "/v1/messages": fails.append("path")
    if body.get("thinking") != {"type": "disabled"}: fails.append(("thinking", body.get("thinking")))
    for k in ("temperature", "top_p", "top_k"):
        if k in body: fails.append(("sent", k))
    if body.get("model") != "claude-haiku-5-5" or body.get("max_tokens") != 16: fails.append("model/max_tokens")
    if body.get("messages") != [{"role": "user", "content": "prompt text"}]: fails.append("messages")
else:
    fails.append("no request captured")
if txt != "LONG" or m.input_tokens != 123 or m.output_tokens != 2 or m.stop_reasons.get("end_turn") != 1:
    fails.append("C16 response fields")
# probe-internal falsifier: misspelled kwarg must be rejected by the SDK
c = anthropic.Anthropic(api_key="test-key-not-real")
try:
    c.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=[{"role": "user", "content": "x"}], thinkng={"type": "disabled"})
    fails.append("typo kwarg accepted")
except TypeError as e:
    print("typo kwarg rejected:", str(e)[:80])
srv.shutdown(); print("fails:", fails); sys.exit(1 if fails else 0)
