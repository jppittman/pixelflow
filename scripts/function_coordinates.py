import os, re, json, collections
ROOT="/home/user/core-term"
ty_re=re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|union|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)')
types=set(); rs=[]
for dp,dirs,files in os.walk(ROOT):
    for d in('target','.git'):
        if d in dirs: dirs.remove(d)
    for f in files:
        if f.endswith('.rs'):
            p=os.path.join(dp,f); rs.append(p)
            for line in open(p,errors='replace'):
                m=ty_re.match(line)
                if m: types.add(m.group(1))
def csnake(s):
    s=re.sub(r'(.)([A-Z][a-z]+)',r'\1_\2',s);s=re.sub(r'([a-z0-9])([A-Z])',r'\1_\2',s);return s.lower()
# COORDINATE vocab = domain/representation/type nouns ("which / for what")
COORD=set()
for t in types:
    for tok in csnake(t).split('_'):
        if len(tok)>=3: COORD.add(tok)
COORD |= {'arena','dag','jet','sh2','simd','scanline','expr','egraph','nnue','hce','field',
 'x86','x64','aarch64','arm','neon','avx','avx512','sse','sse2','scalar','gpu','cpu',
 'tree','graph','node','edge','leaf','class','term','glyph','sdf','ttf','font','rgba','rgb',
 'pty','ansi','csi','sgr','vt','utf8','x11','cocoa','metal','wasm','macos','linux','headless',
 'corpus','replay','trajectory','policy','critic','value','actor','channel','lane','kernel',
 'manifold','lattice','spherical','harmonic','mouse','cursor','selection','viewport','screen'}
# QUALIFIER vocab = adverbs/modifiers ("how / which variant of the action")
QUAL={'with','without','ctx','context','hoisted','parallel','seq','sequential','fast','slow',
 'raw','checked','unchecked','safe','unsafe','internal','inner','impl','default','simple',
 'full','partial','lazy','eager','batch','single','multi','wide','deep','async','sync',
 'mut','ref','owned','borrowed','new','old','legacy','v2','opt','optimized','preserving',
 'incremental','static','dynamic','const'}
# VERBS we expect as the action head (not exhaustive; used to find the head)
VERBS={'compile','emit','encode','decode','parse','eval','evaluate','render','build','make',
 'create','read','write','load','store','save','get','set','find','extract','derive','optimize',
 'handle','process','run','draw','rasterize','sample','collapse','warp','select','map','fold',
 'reduce','forward','backward','backprop','train','update','compute','generate','send','recv',
 'receive','poll','drain','dispatch','translate','convert','format','disassemble','patch',
 'allocate','init','reset','clear','push','pop','insert','remove','add','apply','match','scan',
 'lex','tokenize','classify','annotate','schedule','spawn','step','measure','bench'}

fn_re=re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)')
def crate_of(p): return os.path.relpath(p,ROOT).split(os.sep)[0]
recs=[]
for p in rs:
    crate=crate_of(p)
    for i,line in enumerate(open(p,errors='replace').read().splitlines()):
        m=fn_re.match(line)
        if not m: continue
        name=m.group(1); toks=name.split('_')
        verb = toks[0] if toks[0] in VERBS else None
        rest = toks[1:] if verb else toks
        coords=[t for t in rest if t in COORD]
        quals=[t for t in rest if t in QUAL]
        other=[t for t in rest if t not in COORD and t not in QUAL and len(t)>=2]
        recs.append({'crate':crate,'file':os.path.relpath(p,ROOT),'line':i+1,'name':name,
                     'verb':verb,'coords':coords,'quals':quals,'other':other})
json.dump(recs,open('/tmp/coords.json','w'))
# distribution by # of coordinate tokens
dist=collections.Counter(len(r['coords']) for r in recs)
print("coordinate-token count -> #functions:")
for k in sorted(dist): print(f"  {k}: {dist[k]}")
print("\nfunctions with >=2 coordinate tokens (latent namespace too deep in the name):",
      sum(1 for r in recs if len(r['coords'])>=2))
print("\n=== worst offenders: >=3 coordinate tokens ===")
for r in sorted(recs,key=lambda r:-len(r['coords']))[:25]:
    if len(r['coords'])<3: break
    print(f"  {r['name']:42s} verb={r['verb']}  coords={r['coords']} quals={r['quals']}")
