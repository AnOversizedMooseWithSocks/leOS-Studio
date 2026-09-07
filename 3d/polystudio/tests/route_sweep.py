"""Call EVERY route with a reasonable request and classify what breaks.

The invariant: a route may succeed, or refuse with 4xx because it needs parameters, but it must never
fail *inside the app's own code*. `/api/photo_post` violated that on every request for two releases
and nothing noticed, because nothing had ever called it.

Since `tests/fake_engine/` is deliberately minimal, most routes fail on a missing `holographic_*`
module. Those are gaps in the stub, not app bugs, and are classified out by reading the traceback --
which means this harness stays useful as the stubs grow without needing to be rewritten.

Two real findings came from this sweep:
  * /api/render's ADAPTIVE branch (session= + target_fps=) had never run, even though it is the
    client's DEFAULT preview path -- earlier tests only hit the manual-quality branch.
  * holographic_matlib.classes() is a function, not a dict; the stub had it wrong, which meant
    /api/materials -- the very first call the client makes at boot -- was never exercised.

    python3 tests/route_sweep.py
"""
import io, json, logging, os, re, sys, threading, time, socket, traceback, urllib.error, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, 'fake_engine'))
sys.path.insert(0, os.path.join(ROOT, 'holostuff'))    # flatcompat, as the real launcher provides it
sys.path.insert(0, ROOT)
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
import backend
from flask import Flask

app = Flask(__name__); app.register_blueprint(backend.bp)
app.config['PROPAGATE_EXCEPTIONS'] = False
errors = {}
orig = app.log_exception
def cap(exc_info):
    errors[cap.current] = ''.join(traceback.format_exception(*exc_info))
app.log_exception = cap

s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
threading.Thread(target=lambda: app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False), daemon=True).start()
for _ in range(100):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/api/scene', timeout=1).read(); break
    except Exception: time.sleep(0.05)

scene = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/scene', timeout=5).read())
oid = scene['objects'][0]['id'] if scene.get('objects') else 'o1'
print(f'server up, object id = {oid}\n')

SKIP = {'photo', 'agent/invoke'}                    # streaming / needs a tool name; covered elsewhere
GET_ARGS = f'?object={oid}&name=gold&session=sw&q=fillet&w=120&h=90&res=48&grid=40&spp=8'
POST_BODY = {'object': oid, 'name': 'gold', 'kind': 'cube', 'op': 'noop', 'action': 'list',
             'text': 'sphere radius .5', 'command': 'select all', 'material': 'gold'}

rows = []
for rule in app.url_map.iter_rules():
    path = str(rule)
    if not path.startswith('/api/'): continue
    name = path[len('/api/'):]
    if name in SKIP or '<' in path: continue
    methods = [m for m in rule.methods if m in ('GET', 'POST')]
    for m in sorted(methods):
        cap.current = f'{m} {name}'
        try:
            if m == 'GET':
                r = urllib.request.urlopen(f'http://127.0.0.1:{port}{path}{GET_ARGS}', timeout=25)
            else:
                req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                             data=json.dumps(POST_BODY).encode(), method='POST',
                                             headers={'Content-Type': 'application/json'})
                r = urllib.request.urlopen(req, timeout=25)
            rows.append((m, name, r.status, ''))
        except urllib.error.HTTPError as e:
            body = e.read()[:120]
            rows.append((m, name, e.code, body.decode('utf8', 'replace').replace('\n', ' ')[:100]))
        except Exception as e:
            rows.append((m, name, 'EXC', f'{type(e).__name__}: {e}'[:100]))

ok = [r for r in rows if r[2] == 200]
four = [r for r in rows if isinstance(r[2], int) and 400 <= r[2] < 500]
five = [r for r in rows if r[2] == 500 or r[2] == 'EXC']
print(f'{len(rows)} calls: {len(ok)} ok, {len(four)} 4xx (expected -- needs params), {len(five)} 500/EXC\n')
STUB = re.compile(r"No module named 'holographic|module 'holographic\w*' has no attribute"
                  r"|holographic\w+\.\w+\(\) (?:missing|got)|leCore engine not found|No module named 'flatcompat'")
real, stub = [], []
for m, name, code, msg in five:
    tb = errors.get(f'{m} {name}', '')
    last = tb.strip().split('\n')[-1] if tb else msg
    (stub if STUB.search(tb or msg) else real).append((m, name, last[:150]))
print(f'{len(stub)} fake-engine gaps (expected -- the stub is minimal, not a bug in the app)')
print(f'\n--- {len(real)} FAILED INSIDE THE APP ---')
for m, n, l in sorted(real):
    print(f'  {m:4} {n:24} {l}')
if real:
    print('\nA route failing here has never worked. Investigate before shipping.')
sys.exit(1 if real else 0)
