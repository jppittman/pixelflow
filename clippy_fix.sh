git restore pixelflow-compiler/src/lexer.rs pixelflow-compiler/src/optimize.rs pixelflow-compiler/src/parser.rs pixelflow-compiler/src/symbol.rs pixelflow-compiler/src/annotate.rs pixelflow-compiler/src/ast.rs pixelflow-compiler/src/codegen/emitter.rs pixelflow-compiler/src/codegen/leveled.rs pixelflow-compiler/src/codegen/struct_emitter.rs pixelflow-compiler/src/fold.rs pixelflow-compiler/src/ir_bridge.rs pixelflow-compiler/src/cost_builder.rs
cargo check -p pixelflow-compiler
cargo test -p pixelflow-compiler
