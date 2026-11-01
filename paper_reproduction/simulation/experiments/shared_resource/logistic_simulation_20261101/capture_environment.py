from pathlib import Path
import importlib.util,sys,json,hashlib,platform
R=Path(__file__).resolve().parent;W=R.parents[2]
spec=importlib.util.spec_from_file_location('simulation_code',R/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
paths=[]
for mod in list(sys.modules.values()):
 f=getattr(mod,'__file__',None)
 if f:
  p=Path(f).resolve()
  if p.is_relative_to(W) and p.suffix=='.py' and not any(x in p.parts for x in ['.venv','_deps','.deps']):paths.append(p)
import numpy,scipy,sklearn,matplotlib
obj=dict(python=platform.python_version(),packages={m.__name__:m.__version__ for m in [numpy,scipy,sklearn,matplotlib]},
 files={p.relative_to(W).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})
(R/'environment.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
print(json.dumps(obj,indent=2))
