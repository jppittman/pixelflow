use core_term::config::{Keybinding, KeybindingsConfig, RawKeybindingsConfig};
use core_term::keys::{KeySymbol, Modifiers};
use core_term::term::action::UserInputAction;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_keybindings(c: &mut Criterion) {
    let mut group = c.benchmark_group("keybindings");

    for size in [10, 100, 1000].iter() {
        // Setup config with `size` bindings
        let mut bindings = Vec::new();
        for i in 0..*size {
            // Create dummy bindings
            bindings.push(Keybinding {
                // Use a changing char to avoid duplicates
                key: KeySymbol::Char(char::from_u32(33 + (i % 90) as u32).unwrap_or('a')),
                mods: if i % 2 == 0 {
                    Modifiers::CONTROL
                } else {
                    Modifiers::ALT
                },
                action: UserInputAction::RequestQuit,
            });
        }

        // Add a target binding at the very end to simulate worst-case for linear search
        let target_key = KeySymbol::Enter;
        let target_mods = Modifiers::CONTROL | Modifiers::SHIFT;

        bindings.push(Keybinding {
            key: target_key,
            mods: target_mods,
            action: UserInputAction::InitiateCopy,
        });

        let raw = RawKeybindingsConfig {
            bindings: bindings.clone(),
        };
        let config: KeybindingsConfig = raw.into(); // Populates lookup

        // Benchmark O(1) Map Lookup
        group.bench_function(format!("lookup_map_size_{}", size), |b| {
            b.iter(|| {
                let key = black_box(target_key);
                let mods = black_box(target_mods);
                let _ = config.lookup.get(&(key, mods));
            })
        });

        // Benchmark O(n) Vec Lookup (worst case: it's at the end)
        group.bench_function(format!("lookup_vec_size_{}", size), |b| {
            b.iter(|| {
                let key = black_box(target_key);
                let mods = black_box(target_mods);
                let _ = config
                    .bindings
                    .iter()
                    .find(|b| b.key == key && b.mods == mods)
                    .map(|b| &b.action);
            })
        });

        // Benchmark O(log n) Binary Search
        let mut sorted_bindings = bindings.clone();
        // Requires KeySymbol and Modifiers to implement Ord (added in pixelflow-runtime)
        sorted_bindings.sort_by(|a, b| (a.key, a.mods).cmp(&(b.key, b.mods)));

        group.bench_function(format!("lookup_binsearch_size_{}", size), |b| {
            b.iter(|| {
                let key = black_box(target_key);
                let mods = black_box(target_mods);
                // Note: binary_search_by returns Result<usize, usize>
                let found = sorted_bindings
                    .binary_search_by(|probe| (probe.key, probe.mods).cmp(&(key, mods)));
                black_box(found)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_keybindings);
criterion_main!(benches);
