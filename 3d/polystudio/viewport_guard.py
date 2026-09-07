"""Guard against the class of bug that made the viewport dead: anything stacked over the 3D canvas that
does not let pointer events through. Static, so it runs anywhere -- no browser needed."""
import re, sys
h=open('index.html').read(); j=open('app.js').read()
fails=[]
# 1) dynamically created full-cover elements must be click-through (concatenated literals joined first)
for m in re.finditer(r"style\.cssText\s*=\s*((?:'[^']*'\s*\+?\s*)+)", j):
    css=''.join(re.findall(r"'([^']*)'", m.group(1))).replace(' ','')
    if ('inset:0' in css or ('width:100%' in css and 'height:100%' in css)) and 'position:absolute' in css:
        if 'pointer-events:none' not in css and 'display:none' not in css.split('pointer-events')[0][:0]+css:
            pass
        if 'pointer-events:none' not in css:
            fails.append('dynamic overlay without pointer-events:none -> %s'%css[:70])
# 2) the GPU overlay specifically (it is shown, not just created)
gpu=re.search(r"id='gpucanvas'.*?cssText\s*=\s*((?:'[^']*'\s*\+?\s*)+)", j, re.S)
if gpu:
    css=''.join(re.findall(r"'([^']*)'", gpu.group(1))).replace(' ','')
    if 'pointer-events:none' not in css:
        fails.append('#gpucanvas overlay would swallow viewport input')
# 3) nothing may auto-enable the GPU path at load
if re.search(r"navigator\.gpu[\s\S]{0,400}?GPU\.on\s*=\s*true", j):
    fails.append('GPU preview auto-enables at startup (must be opt-in)')
# 4) app.js must not be reachable before THREE is confirmed
if re.search(r'<script src="app\.js"></script>', h):
    fails.append('app.js loaded directly; must load only after the THREE chain resolves')
print('VIEWPORT INPUT GUARD:', 'PASS' if not fails else 'FAIL')
for f in fails: print('  *', f)
sys.exit(1 if fails else 0)
