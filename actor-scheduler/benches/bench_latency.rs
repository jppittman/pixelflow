use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

struct LatencyActor {
    response_tx: mpsc::Sender<()>,
}

impl Actor<(), (), ()> for LatencyActor {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }

    fn handle_control(&mut self, _: ()) -> HandlerResult {
        // Immediately signal completion
        self.response_tx.send(()).ok();
        Ok(())
    }

    fn handle_management(&mut self, _: ()) -> HandlerResult {
        // Immediately signal completion
        self.response_tx.send(()).ok();
        Ok(())
    }

    fn park(&mut self, hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
        match hint {
            SystemStatus::Idle => Ok(ActorStatus::Idle),
            SystemStatus::Busy => Ok(ActorStatus::Busy),
        }
    }
}

fn bench_control_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_latency_steady_state");

    for buffer_size in [10, 32, 100] {
        group.bench_with_input(
            BenchmarkId::new("buffer", buffer_size),
            &buffer_size,
            |b, &size| {
                // Spawn actor thread ONCE, outside the benchmark loop
                let (response_tx, response_rx) = mpsc::channel();
                let (tx, mut rx) = ActorScheduler::new(1024, size);

                let actor_handle = thread::spawn(move || {
                    let mut actor = LatencyActor { response_tx };
                    rx.run(&mut actor);
                });

                // Give actor thread time to start and park
                thread::sleep(Duration::from_millis(1));

                // Benchmark individual message round-trips (steady-state)
                b.iter(|| {
                    let start = Instant::now();
                    tx.send(Message::Control(())).unwrap();
                    response_rx.recv().unwrap();
                    let latency = start.elapsed();
                    black_box(latency)
                });

                // Cleanup
                tx.send(Message::Shutdown).unwrap();
                actor_handle.join().unwrap();
            },
        );
    }
    group.finish();
}

fn bench_control_latency_under_load(c: &mut Criterion) {
    c.bench_function("control_latency_under_data_flood", |b| {
        // Spawn actor thread ONCE
        let (response_tx, response_rx) = mpsc::channel();
        let mut builder = ActorBuilder::<(), (), ()>::new(100, None);
        let tx = builder.add_producer();
        let tx_data = builder.add_producer();
        let mut rx = builder.build();

        let actor_handle = thread::spawn(move || {
            let mut actor = LatencyActor { response_tx };
            rx.run(&mut actor);
        });

        thread::sleep(Duration::from_millis(1));

        // Start continuous data flooder
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let flooder = thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                tx_data.send(Message::Data(())).ok();
            }
        });

        // Let flood establish
        thread::sleep(Duration::from_millis(10));

        // Benchmark control latency while data is continuously flooding
        b.iter(|| {
            let start = Instant::now();
            tx.send(Message::Control(())).unwrap();
            response_rx.recv().unwrap();
            let latency = start.elapsed();
            black_box(latency)
        });

        // Cleanup
        stop.store(true, Ordering::Relaxed);
        flooder.join().unwrap();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
    });
}

fn bench_management_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("management_latency_steady_state");

    for buffer_size in [10, 32, 100] {
        group.bench_with_input(
            BenchmarkId::new("buffer", buffer_size),
            &buffer_size,
            |b, &size| {
                // Spawn actor thread ONCE
                let (response_tx, response_rx) = mpsc::channel();
                let (tx, mut rx) = ActorScheduler::new(1024, size);

                let actor_handle = thread::spawn(move || {
                    let mut actor = LatencyActor { response_tx };
                    rx.run(&mut actor);
                });

                thread::sleep(Duration::from_millis(1));

                // Benchmark individual management message round-trips
                b.iter(|| {
                    let start = Instant::now();
                    tx.send(Message::Management(())).unwrap();
                    response_rx.recv().unwrap();
                    let latency = start.elapsed();
                    black_box(latency)
                });

                // Cleanup
                tx.send(Message::Shutdown).unwrap();
                actor_handle.join().unwrap();
            },
        );
    }
    group.finish();
}

fn bench_management_latency_under_load(c: &mut Criterion) {
    c.bench_function("management_latency_under_control_flood", |b| {
        // Spawn actor thread ONCE
        let (response_tx, response_rx) = mpsc::channel();
        let mut builder = ActorBuilder::<(), (), ()>::new(100, None);
        let tx = builder.add_producer();
        let tx_control = builder.add_producer();
        let mut rx = builder.build();

        let actor_handle = thread::spawn(move || {
            let mut actor = LatencyActor { response_tx };
            rx.run(&mut actor);
        });

        thread::sleep(Duration::from_millis(1));

        // Start continuous control flooder
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let flooder = thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                tx_control.send(Message::Control(())).ok();
            }
        });

        // Let flood establish
        thread::sleep(Duration::from_millis(10));

        // Benchmark management latency while control is continuously flooding
        b.iter(|| {
            let start = Instant::now();
            tx.send(Message::Management(())).unwrap();
            response_rx.recv().unwrap();
            let latency = start.elapsed();
            black_box(latency)
        });

        // Cleanup
        stop.store(true, Ordering::Relaxed);
        flooder.join().unwrap();
        tx.send(Message::Shutdown).unwrap();
        actor_handle.join().unwrap();
    });
}

criterion_group!(
    benches,
    bench_control_latency,
    bench_control_latency_under_load,
    bench_management_latency,
    bench_management_latency_under_load
);
criterion_main!(benches);
