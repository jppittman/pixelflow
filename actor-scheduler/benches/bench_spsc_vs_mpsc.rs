//! Head-to-head comparison: std::sync::mpsc vs sharded SPSC.
//!
//! Tests the same workloads through both channel backends to validate
//! the performance hypothesis before migrating consumers.
//!
//! Run: cargo bench -p actor-scheduler --bench bench_spsc_vs_mpsc

use actor_scheduler::sharded::InboxBuilder;
use actor_scheduler::spsc;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TrySendError};
use std::thread;

// ─── Single-producer throughput ───────────────────────────────────────────

fn bench_single_producer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_producer_throughput");

    for count in [10_000, 100_000, 1_000_000] {
        // MPSC baseline
        group.bench_with_input(BenchmarkId::new("mpsc", count), &count, |b, &count| {
            b.iter(|| {
                let (tx, rx) = mpsc::sync_channel::<u64>(1024);
                let consumer = thread::spawn(move || {
                    let mut n = 0u64;
                    loop {
                        match rx.try_recv() {
                            Ok(v) => n += v,
                            Err(mpsc::TryRecvError::Empty) => thread::yield_now(),
                            Err(mpsc::TryRecvError::Disconnected) => break,
                        }
                    }
                    n
                });

                for i in 0..count {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(TrySendError::Full(_)) => thread::yield_now(),
                            Err(TrySendError::Disconnected(_)) => panic!(),
                        }
                    }
                }
                drop(tx);
                black_box(consumer.join().unwrap());
            });
        });

        // SPSC
        group.bench_with_input(BenchmarkId::new("spsc", count), &count, |b, &count| {
            b.iter(|| {
                let (mut tx, mut rx) = spsc::spsc_channel::<u64>(1024);
                let consumer = thread::spawn(move || {
                    let mut n = 0u64;
                    loop {
                        match rx.try_recv() {
                            Ok(v) => n += v,
                            Err(spsc::TryRecvError::Empty) => thread::yield_now(),
                            Err(spsc::TryRecvError::Disconnected) => break,
                        }
                    }
                    n
                });

                for i in 0..count {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(spsc::TrySendError::Full(_)) => thread::yield_now(),
                            Err(spsc::TrySendError::Disconnected(_)) => panic!(),
                        }
                    }
                }
                drop(tx);
                black_box(consumer.join().unwrap());
            });
        });
    }

    group.finish();
}

// ─── Multi-producer throughput (the real test) ────────────────────────────

fn bench_multi_producer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_producer_throughput");

    for num_producers in [2, 4, 8] {
        let msg_per_producer = 50_000u64;

        // MPSC: N producers clone the same SyncSender
        group.bench_with_input(
            BenchmarkId::new("mpsc", num_producers),
            &num_producers,
            |b, &n_prod| {
                b.iter(|| {
                    let (tx, rx) = mpsc::sync_channel::<u64>(1024);

                    let consumer = thread::spawn(move || {
                        let mut total = 0u64;
                        loop {
                            match rx.try_recv() {
                                Ok(v) => total += v,
                                Err(mpsc::TryRecvError::Empty) => thread::yield_now(),
                                Err(mpsc::TryRecvError::Disconnected) => break,
                            }
                        }
                        total
                    });

                    let producers: Vec<_> = (0..n_prod)
                        .map(|_| {
                            let tx = tx.clone();
                            thread::spawn(move || {
                                for i in 0..msg_per_producer {
                                    loop {
                                        match tx.try_send(i) {
                                            Ok(()) => break,
                                            Err(TrySendError::Full(_)) => thread::yield_now(),
                                            Err(TrySendError::Disconnected(_)) => return,
                                        }
                                    }
                                }
                            })
                        })
                        .collect();

                    drop(tx); // drop original so consumer sees disconnect
                    for p in producers {
                        p.join().unwrap();
                    }
                    black_box(consumer.join().unwrap());
                });
            },
        );

        // Sharded SPSC: N producers, each with own SPSC channel
        group.bench_with_input(
            BenchmarkId::new("sharded_spsc", num_producers),
            &num_producers,
            |b, &n_prod| {
                b.iter(|| {
                    let mut builder = InboxBuilder::<u64>::new(1024);
                    let senders: Vec<_> = (0..n_prod).map(|_| builder.add_producer()).collect();
                    let mut inbox = builder.build();

                    let consumer = thread::spawn(move || {
                        let mut total = 0u64;
                        loop {
                            match inbox.drain(256, |v| {
                                total += v;
                                Ok(())
                            }) {
                                Ok(actor_scheduler::sharded::DrainStatus::Disconnected) => break,
                                Ok(_) => {}
                                Err(_) => break,
                            }
                            thread::yield_now();
                        }
                        total
                    });

                    let producers: Vec<_> = senders
                        .into_iter()
                        .map(|mut tx| {
                            thread::spawn(move || {
                                for i in 0..msg_per_producer {
                                    loop {
                                        match tx.try_send(i) {
                                            Ok(()) => break,
                                            Err(spsc::TrySendError::Full(_)) => thread::yield_now(),
                                            Err(spsc::TrySendError::Disconnected(_)) => return,
                                        }
                                    }
                                }
                            })
                        })
                        .collect();

                    for p in producers {
                        p.join().unwrap();
                    }
                    black_box(consumer.join().unwrap());
                });
            },
        );
    }

    group.finish();
}

// ─── Send-side latency (the wait-free claim) ──────────────────────────────

fn bench_send_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("send_latency_ns");

    // MPSC: single send with a background consumer draining
    group.bench_function("mpsc", |b| {
        let (tx, rx) = mpsc::sync_channel::<u64>(4096);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = stop.clone();
        let _consumer = thread::spawn(move || {
            while !stop_c.load(Ordering::Relaxed) {
                rx.try_recv().ok();
            }
        });
        thread::sleep(std::time::Duration::from_millis(1));

        let mut i = 0u64;
        b.iter(|| {
            tx.try_send(i).ok();
            i += 1;
            black_box(i);
        });
        stop.store(true, Ordering::Relaxed);
    });

    // SPSC: single send with a background consumer draining
    group.bench_function("spsc", |b| {
        let (mut tx, mut rx) = spsc::spsc_channel::<u64>(4096);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = stop.clone();
        let _consumer = thread::spawn(move || {
            while !stop_c.load(Ordering::Relaxed) {
                rx.try_recv().ok();
            }
        });
        thread::sleep(std::time::Duration::from_millis(1));

        let mut i = 0u64;
        b.iter(|| {
            tx.try_send(i).ok();
            i += 1;
            black_box(i);
        });
        stop.store(true, Ordering::Relaxed);
    });

    group.finish();
}

// ─── Send latency under contention (N producers racing) ──────────────────

fn bench_send_latency_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("send_latency_contended");

    for num_contenders in [2, 4] {
        // MPSC: other threads flooding the same channel
        group.bench_with_input(
            BenchmarkId::new("mpsc", num_contenders),
            &num_contenders,
            |b, &n| {
                let (tx, rx) = mpsc::sync_channel::<u64>(4096);
                let stop = Arc::new(AtomicBool::new(false));

                // Background consumer
                let _consumer = {
                    let stop = stop.clone();
                    thread::spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            rx.try_recv().ok();
                            thread::yield_now();
                        }
                    })
                };

                // Background contenders
                let _contenders: Vec<_> = (0..n)
                    .map(|_| {
                        let tx = tx.clone();
                        let stop = stop.clone();
                        thread::spawn(move || {
                            let mut i = 0u64;
                            while !stop.load(Ordering::Relaxed) {
                                tx.try_send(i).ok();
                                i += 1;
                            }
                        })
                    })
                    .collect();

                thread::sleep(std::time::Duration::from_millis(5));

                let mut i = 0u64;
                b.iter(|| {
                    tx.try_send(i).ok();
                    i += 1;
                    black_box(i);
                });

                stop.store(true, Ordering::Relaxed);
                // Contenders and consumer will stop on next loop iteration
            },
        );

        // Sharded SPSC: each contender has its own channel — NO contention
        group.bench_with_input(
            BenchmarkId::new("sharded_spsc", num_contenders),
            &num_contenders,
            |b, &n| {
                let mut builder = InboxBuilder::<u64>::new(4096);
                let mut my_tx = builder.add_producer();
                let contender_txs: Vec<_> = (0..n).map(|_| builder.add_producer()).collect();
                let mut inbox = builder.build();

                let stop = Arc::new(AtomicBool::new(false));

                // Background consumer
                let stop_c = stop.clone();
                let _consumer = thread::spawn(move || {
                    while !stop_c.load(Ordering::Relaxed) {
                        inbox.drain(256, |_v: u64| Ok(())).ok();
                        thread::yield_now();
                    }
                });

                // Background contenders — each with own SPSC (no shared state)
                let _contenders: Vec<_> = contender_txs
                    .into_iter()
                    .map(|mut tx| {
                        let stop = stop.clone();
                        thread::spawn(move || {
                            let mut i = 0u64;
                            while !stop.load(Ordering::Relaxed) {
                                tx.try_send(i).ok();
                                i += 1;
                            }
                        })
                    })
                    .collect();

                thread::sleep(std::time::Duration::from_millis(5));

                let mut i = 0u64;
                b.iter(|| {
                    my_tx.try_send(i).ok();
                    i += 1;
                    black_box(i);
                });

                stop.store(true, Ordering::Relaxed);
            },
        );
    }

    group.finish();
}

// ─── Round-trip latency (send → process → ack) ───────────────────────────

fn bench_roundtrip_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_latency");

    // MPSC round-trip
    group.bench_function("mpsc", |b| {
        let (tx, rx) = mpsc::sync_channel::<u64>(64);
        let (ack_tx, ack_rx) = mpsc::sync_channel::<()>(1);

        let _worker = thread::spawn(move || {
            loop {
                match rx.recv() {
                    Ok(_v) => {
                        ack_tx.try_send(()).ok();
                    }
                    Err(_) => break,
                }
            }
        });

        thread::sleep(std::time::Duration::from_millis(1));

        let mut i = 0u64;
        b.iter(|| {
            tx.send(i).unwrap();
            ack_rx.recv().unwrap();
            i += 1;
            black_box(i);
        });
    });

    // SPSC round-trip
    group.bench_function("spsc", |b| {
        let (mut tx, mut rx) = spsc::spsc_channel::<u64>(64);
        let (mut ack_tx, mut ack_rx) = spsc::spsc_channel::<()>(1);

        let _worker = thread::spawn(move || {
            loop {
                match rx.try_recv() {
                    Ok(_v) => loop {
                        match ack_tx.try_send(()) {
                            Ok(()) => break,
                            Err(spsc::TrySendError::Full(_)) => thread::yield_now(),
                            Err(spsc::TrySendError::Disconnected(_)) => return,
                        }
                    },
                    Err(spsc::TryRecvError::Empty) => thread::yield_now(),
                    Err(spsc::TryRecvError::Disconnected) => break,
                }
            }
        });

        thread::sleep(std::time::Duration::from_millis(1));

        let mut i = 0u64;
        b.iter(|| {
            loop {
                match tx.try_send(i) {
                    Ok(()) => break,
                    Err(spsc::TrySendError::Full(_)) => thread::yield_now(),
                    Err(spsc::TrySendError::Disconnected(_)) => panic!(),
                }
            }
            loop {
                match ack_rx.try_recv() {
                    Ok(()) => break,
                    Err(spsc::TryRecvError::Empty) => thread::yield_now(),
                    Err(spsc::TryRecvError::Disconnected) => panic!(),
                }
            }
            i += 1;
            black_box(i);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_producer_throughput,
    bench_multi_producer_throughput,
    bench_send_latency,
    bench_send_latency_contended,
    bench_roundtrip_latency,
);
criterion_main!(benches);
