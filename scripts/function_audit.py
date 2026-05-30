import os, re, json, collections, csv
ROOT="/home/user/core-term"
fn_re=re.compile(r'^(?P<ind>\s*)(?P<vis>pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)')
impl_start=re.compile(r'^\s*(?:unsafe\s+)?impl(?:\s*<.*)?\b')
impl_for=re.compile(r'\bfor\s+(?P<ty>[A-Za-z_][A-Za-z0-9_:]*)')
impl_self=re.compile(r'^\s*(?:unsafe\s+)?impl(?:\s*<[^>]*>)?\s+(?P<ty>[A-Za-z_][A-Za-z0-9_:]*)')
trait_start=re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?trait\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)')
mod_start=re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)')
def crate_of(p): return os.path.relpath(p,ROOT).split(os.sep)[0]
records=[]
for dp,dirs,files in os.walk(ROOT):
    for d in ('target','.git'):
        if d in dirs: dirs.remove(d)
    for f in files:
        if not f.endswith('.rs'): continue
        path=os.path.join(dp,f); crate=crate_of(path)
        try: lines=open(path,encoding='utf-8',errors='replace').read().splitlines()
        except: continue
        stack=[]; pending=None; depth=0; attrs=[]
        is_test_file='/tests/' in path or '/benches/' in path
        for i,line in enumerate(lines):
            s=line.strip()
            if s.startswith('#['): attrs.append(s)
            m=fn_re.match(line)
            if m:
                name=m.group('name'); vis=(m.group('vis') or '').strip()
                ck,cn='free',None
                for k,nm,d in reversed(stack):
                    if k in('impl','trait'): ck,cn=k,nm;break
                in_test=any(k=='testmod' for k,_,_ in stack)
                ta=any(a.startswith('#[test')or a.startswith('#[bench')or'tokio::test'in a for a in attrs)
                records.append({'crate':crate,'file':os.path.relpath(path,ROOT),'line':i+1,'name':name,
                  'vis':'pub' if vis.startswith('pub')and'('not in vis else('pub(crate)'if'crate'in vis else('pub(r)'if vis.startswith('pub')else'priv')),
                  'ctx':ck,'ctx_name':cn or'',
                  'test':is_test_file or in_test or ta or name.startswith(('test_','bench_'))})
            if pending is None:
                if impl_start.match(line):
                    mf=impl_for.search(line);ms=impl_self.match(line)
                    ty=(mf.group('ty')if mf else(ms.group('ty')if ms else'?')).split('<')[0]
                    pending=('impl',ty)
                elif trait_start.match(line): pending=('trait',trait_start.match(line).group('name'))
                elif mod_start.match(line):
                    nm=mod_start.match(line).group('name'); ct=any('cfg(test)'in a for a in attrs)
                    pending=('testmod'if(ct or nm=='tests')else'mod',nm)
            if not s.startswith('#['): attrs=[]
            o=line.count('{');c=line.count('}')
            if pending and o>0: stack.append((pending[0],pending[1],depth));pending=None
            depth+=o;depth-=c
            while stack and depth<=stack[-1][2]: stack.pop()
# write full dump TSV
recs=sorted(records,key=lambda r:(r['crate'],r['file'],r['line']))
with open(os.path.join(ROOT,'docs/function-audit.tsv'),'w',newline='') as fh:
    w=csv.writer(fh,delimiter='\t')
    w.writerow(['crate','file','line','name','visibility','context','context_type','is_test'])
    for r in recs:
        w.writerow([r['crate'],r['file'],r['line'],r['name'],r['vis'],r['ctx'],r['ctx_name'],'test'if r['test']else'prod'])
json.dump(records,open('/tmp/fns_final.json','w'))
print("wrote docs/function-audit.tsv  rows:",len(recs))
print("prod:",sum(1 for r in records if not r['test']),"test:",sum(1 for r in records if r['test']))
