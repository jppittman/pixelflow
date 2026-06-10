use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::Duration;

struct CountingActor {
    data_count: Arc<AtomicUsize>,
    control_count: Arc<AtomicUsize>,
}

impl Actor<i32, (), ()> for CountingActor {
    fn handle_data(&mut self, _data: i32) -> HandlerResult {
        self.data_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn handle_control(&mut self, _ctrl: ()) -> HandlerResult {
        self.control_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn handle_management(&mut self, _mgmt: ()) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        match hint {
            SystemStatus::Idle => Ok(ActorStatus::Idle),
            SystemStatus::Busy => Ok(ActorStatus::Busy),
        }
    }
}

fn bench_control_flood_impact(c: &mut Criterion) {
    c.bench_function("data_throughput_under_control_flood", |b| {
        b.iter(|| {
            let data_count = Arc::new(AtomicUsize::new(0));
            let control_count = Arc::new(AtomicUsize::new(0));

            let mut builder = ActorBuilder::<i32, (), ()>::new(1000, None);
            let tx = builder.add_producer();
            let tx_control = builder.add_producer();
            let tx_data = builder.add_producer();
            let mut rx = builder.build();

            let actor_data = data_count.clone();
            let actor_control = control_count.clone();

            let actor_handle = thread::spawn(move || {
                let mut actor = CountingActor {
                    data_count: actor_data,
                    control_count: actor_control,
                };
                rx.run(&mut actor);
            });

            // Start continuous control flooder
            let stop_flooding = Arc::new(AtomicBool::new(false));
            let stop_flag = stop_flooding.clone();
            let control_flooder = thread::spawn(move || {
                let mut sent = 0;
                while !stop_flag.load(Ordering::Relaxed) {
                    if tx_control.send(Message::Control(())).is_ok() {
                        sent += 1;
                    }
                }
                sent
            });

            // Let flooder establish
            thread::sleep(Duration::from_millis(10));

            // Now send data messages while control is flooding
            let data_sender = thread::spawn(move || {
                for i in 0..1000 {
                    tx_data.send(Message::Data(i)).ok();
                }
            });

            data_sender.join().unwrap();

            // Give time for processing
            thread::sleep(Duration::from_millis(50));

            // Stop flooder
            stop_flooding.store(true, Ordering::Relaxed);
            let control_sent = control_flooder.join().unwrap();

            tx.send(Message::Shutdown).unwrap();
            actor_handle.join().unwrap();

            let data_processed = data_count.load(Ordering::Relaxed);
            black_box((data_processed, control_sent))
        });
    });
}

fn bench_burst_limiting_effectiveness(c: &mut Criterion) {
    c.bench_function("burst_limit_vs_unlimited", |b| {
        b.iter(|| {
            let data_count = Arc::new(AtomicUsize::new(0));
            let control_count = Arc::new(AtomicUsize::new(0));

            let (tx, mut rx) = ActorScheduler::new(1024, 100);

            let actor_data = data_count.clone();
            let actor_control = control_count.clone();

            let actor_handle = thread::spawn(move || {
                let mut actor = CountingActor {
                    data_count: actor_data,
                    control_count: actor_control,
                };
                rx.run(&mut actor);
            });

            // Send large batch of control then data
            for _ in 0..1000 {
                tx.send(Message::Control(())).ok();
            }
            for i in 0..100 {
                tx.send(Message::Data(i)).ok();
            }

            thread::sleep(Duration::from_millis(50));

            tx.send(Message::Shutdown).unwrap();
            actor_handle.join().unwrap();

            let data_processed = data_count.load(Ordering::Relaxed);
            let control_processed = control_count.load(Ordering::Relaxed);
            black_box((data_processed, control_processed))
        });
    });
}

fn bench_multiple_control_flooders(c: &mut Criterion) {
    c.bench_function("four_control_flooders_vs_data", |b| {
        b.iter(|| {
            let data_count = Arc::new(AtomicUsize::new(0));
            let control_count = Arc::new(AtomicUsize::new(0));

            let mut builder = ActorBuilder::<i32, (), ()>::new(1000, None);
            let tx = builder.add_producer();
            let flooder_txs: Vec<_> = (0..4).map(|_| builder.add_producer()).collect();
            let tx_data = builder.add_producer();
            let mut rx = builder.build();

            let actor_data = data_count.clone();
            let actor_control = control_count.clone();

            let actor_handle = thread::spawn(move || {
                let mut actor = CountingActor {
                    data_count: actor_data,
                    control_count: actor_control,
                };
                rx.run(&mut actor);
            });

            // Start 4 continuous control flooders
            let stop_flooding = Arc::new(AtomicBool::new(false));
            let mut flooders = vec![];

            for tx_control in flooder_txs {
                let stop_flag = stop_flooding.clone();
                let flooder = thread::spawn(move || {
                    let mut sent = 0;
                    while !stop_flag.load(Ordering::Relaxed) {
                        if tx_control.send(Message::Control(())).is_ok() {
                            sent += 1;
                        }
                    }
                    sent
                });
                flooders.push(flooder);
            }

            // Let flooders establish
            thread::sleep(Duration::from_millis(10));

            // Send data while control is being flooded by 4 threads
            let data_sender = thread::spawn(move || {
                for i in 0..500 {
                    tx_data.send(Message::Data(i)).ok();
                }
            });

            data_sender.join().unwrap();

            // Give time for processing
            thread::sleep(Duration::from_millis(50));

            // Stop all flooders
            stop_flooding.store(true, Ordering::Relaxed);
            let total_control_sent: usize = flooders.into_iter().map(|f| f.join().unwrap()).sum();

            tx.send(Message::Shutdown).unwrap();
            actor_handle.join().unwrap();

            let data_processed = data_count.load(Ordering::Relaxed);
            black_box((data_processed, total_control_sent))
        });
    });
}

fn bench_slow_receiver_backpressure(c: &mut Criterion) {
    c.bench_function("slow_receiver_backpressure", |b| {
        b.iter(|| {
            let data_count = Arc::new(AtomicUsize::new(0));
            let control_count = Arc::new(AtomicUsize::new(0));

            let (tx, mut rx) = ActorScheduler::new(10, 10); // Small buffers

            let actor_data = data_count.clone();
            let actor_control = control_count.clone();

            // Slow actor that sleeps on each message
            struct SlowActor {
                data_count: Arc<AtomicUsize>,
                control_count: Arc<AtomicUsize>,
            }

            impl Actor<i32, (), ()> for SlowActor {
                fn handle_data(&mut self, _data: i32) -> HandlerResult {
                    thread::sleep(Duration::from_micros(100));
                    self.data_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }

                fn handle_control(&mut self, _ctrl: ()) -> HandlerResult {
                    self.control_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }

                fn handle_management(&mut self, _mgmt: ()) -> HandlerResult {
                    Ok(())
                }

                fn park(&mut self, hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
                    match hint {
                        SystemStatus::Idle => Ok(ActorStatus::Idle),
                        SystemStatus::Busy => Ok(ActorStatus::Busy),
                    }
                }
            }

            let actor_handle = thread::spawn(move || {
                let mut actor = SlowActor {
                    data_count: actor_data,
                    control_count: actor_control,
                };
                rx.run(&mut actor);
            });

            // Try to send 100 data messages - should hit backpressure
            let mut sent = 0;
            for i in 0..100 {
                if tx.send(Message::Data(i)).is_ok() {
                    sent += 1;
                }
            }

            thread::sleep(Duration::from_millis(50));

            tx.send(Message::Shutdown).unwrap();
            actor_handle.join().unwrap();

            let processed = data_count.load(Ordering::Relaxed);
            black_box((sent, processed))
        });
    });
}

criterion_group!(
    benches,
    bench_control_flood_impact,
    bench_burst_limiting_effectiveness,
    bench_multiple_control_flooders,
    bench_slow_receiver_backpressure
);
criterion_main!(benches);
