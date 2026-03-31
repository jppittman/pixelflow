with open("pixelflow-compiler/src/annotate.rs", "r") as f:
    text = f.read()
text = text.replace("AnnotatedStmt::Let(AnnotatedLet {", "AnnotatedStmt::Let(Box::new(AnnotatedLet {")
text = text.replace("span: let_stmt.span,\n                }),", "span: let_stmt.span,\n                })),")
with open("pixelflow-compiler/src/annotate.rs", "w") as f:
    f.write(text)

with open("pixelflow-compiler/src/parser.rs", "r") as f:
    text = f.read()
text = text.replace("ParamKind::Scalar(ty)", "ParamKind::Scalar(Box::new(ty))")
text = text.replace("BinaryOp::from_syn(expr_binary.op)", "BinaryOp::from_syn(&expr_binary.op)")
text = text.replace("UnaryOp::from_syn(expr_unary.op)", "UnaryOp::from_syn(&expr_unary.op)")
text = text.replace("stmts.push(Stmt::Let(LetStmt {", "stmts.push(Stmt::Let(Box::new(LetStmt {")
text = text.replace("span: Span::call_site(),\n                }));", "span: Span::call_site(),\n                })));")
with open("pixelflow-compiler/src/parser.rs", "w") as f:
    f.write(text)

with open("pixelflow-compiler/src/optimize.rs", "r") as f:
    text = f.read()
text = text.replace("stmts.push(Stmt::Let(LetStmt {", "stmts.push(Stmt::Let(Box::new(LetStmt {")
text = text.replace("span,\n                }));", "span,\n                })));")
with open("pixelflow-compiler/src/optimize.rs", "w") as f:
    f.write(text)

with open("pixelflow-compiler/src/sema.rs", "r") as f:
    text = f.read()
text = text.replace("register_parameter(param.name.clone(), ty.clone())", "register_parameter(param.name.clone(), *ty.clone())")
with open("pixelflow-compiler/src/sema.rs", "w") as f:
    f.write(text)
