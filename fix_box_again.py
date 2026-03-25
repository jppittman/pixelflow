import os

file = 'pixelflow-compiler/src/optimize.rs'
with open(file, 'r') as f: content = f.read()

content = content.replace('stmts.push(Stmt::Let(LetStmt {', 'stmts.push(Stmt::Let(Box::new(LetStmt {')
content = content.replace('                    span,\n                }));', '                    span,\n                })));')

with open(file, 'w') as f: f.write(content)
