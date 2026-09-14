"""Find the _tonemap bug class: a top-level function referencing a name that only exists
inside SOME OTHER function's body. Python resolves those at call time, so they parse, import
and lint clean -- and then raise NameError the first time the route is hit."""
import ast, builtins, sys

import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SRC = open(os.path.join(ROOT, 'backend.py')).read()
tree = ast.parse(SRC)
BUILTINS = set(dir(builtins))

# every name bound at module level (functions, classes, assignments, imports)
module_names = set()
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        module_names.add(n.name)
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name): module_names.add(x.id)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: module_names.add((a.asname or a.name).split('.')[0])
    elif isinstance(n, ast.If):                       # module-level conditionals
        for sub in ast.walk(n):
            if isinstance(sub, (ast.FunctionDef, ast.ClassDef)): module_names.add(sub.name)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store): module_names.add(sub.id)

# names defined INSIDE some function (nested defs) -- the trap
nested_defs = {}
for top in tree.body:
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
    for sub in ast.walk(top):
        if sub is top: continue
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name not in module_names:
            nested_defs.setdefault(sub.name, set()).add(top.name)

def local_names(fn):
    """Everything bound within fn: params, assignments, imports, comprehension vars, nested defs."""
    out = set()
    for a in list(fn.args.args)+list(fn.args.kwonlyargs)+list(fn.args.posonlyargs):
        out.add(a.arg)
    if fn.args.vararg: out.add(fn.args.vararg.arg)
    if fn.args.kwarg: out.add(fn.args.kwarg.arg)
    for sub in ast.walk(fn):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not fn: out.add(sub.name)
        elif isinstance(sub, ast.ClassDef): out.add(sub.name)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store): out.add(sub.id)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for a in sub.names: out.add((a.asname or a.name).split('.')[0])
        elif isinstance(sub, (ast.ExceptHandler,)) and sub.name: out.add(sub.name)
        elif isinstance(sub, ast.arg): out.add(sub.arg)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)): out.update(sub.names)
    return out

# which top-level functions are Flask routes (highest blast radius)
routes = {}
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in n.decorator_list:
            s = ast.unparse(d)
            if 'bp.route' in s: routes[n.name] = s

problems = []
for top in tree.body:
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
    known = local_names(top) | module_names | BUILTINS
    for sub in ast.walk(top):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id in known: continue
            if sub.id in nested_defs and top.name not in nested_defs[sub.id]:
                problems.append((top.name, sub.id, sub.lineno, sorted(nested_defs[sub.id])))

print(f"scanned {len(routes)} routes, {sum(1 for n in tree.body if isinstance(n, ast.FunctionDef))} top-level functions")
if not problems:
    print("no cross-scope references found")
for fn, name, line, owners in sorted(set((a,b,c,tuple(d)) for a,b,c,d in problems)):
    tag = "ROUTE " + routes[fn] if fn in routes else "func"
    print(f"  backend.py:{line}  {fn}() references '{name}' -- only defined inside {', '.join(owners)}()   [{tag}]")
sys.exit(1 if problems else 0)
