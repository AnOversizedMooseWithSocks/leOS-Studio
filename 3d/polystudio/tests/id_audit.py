# Every $('id') app.js reaches for must exist in index.html (or be guarded).
import re,sys,os
root=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
html=open(os.path.join(root,'index.html')).read()
js=open(os.path.join(root,'app.js')).read()
have=set(re.findall(r'id="([^"]+)"',html))
# ids created at runtime by JS templates count as present
have |= set(re.findall(r"""id=['"]([A-Za-z0-9_\-]+)['"]""", js))
have |= set(re.findall(r"""id='([A-Za-z0-9_\-]+)'""", js))
used=set(re.findall(r"""\$\('([^']+)'\)""", js))
missing=sorted(u for u in used if u not in have)
# a reference is safe if it is guarded ( $('x') && ... )
def guarded(m):
    tok="$('%s')"%m
    return any(pat in js for pat in (tok+" &&", tok+"&&", "&& "+tok, "&&"+tok, "if("+tok+")"))
unguarded=[m for m in missing if not guarded(m)]
print('referenced ids missing from index.html:', len(missing))
for m in missing: print('   ', m, '(guarded)' if m not in unguarded else '  <-- UNGUARDED')
sys.exit(1 if unguarded else 0)
