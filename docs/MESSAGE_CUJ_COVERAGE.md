> **Historical — a coverage plan whose test files were never created under these names
> (annotated 2026-09-09).** `core-term/tests/e2e_message_tests.rs`,
> `engine_app_message_tests.rs` and `pixelflow-runtime/tests/vsync_cuj_tests.rs`
> are not in the tree, and the actor layer has been rewritten twice since
> (supervisor → Mealy transducer, mediator → mesh). The CUJ enumeration is still a
> reasonable statement of *what* the message flows are; the file plan is not a
> description of current tests. Live actor design:
> [`designs/actor-scheduler-mealy-transducer.md`](designs/actor-scheduler-mealy-transducer.md).

# Message CUJ Coverage Plan

## Overview

This document defines the Critical User Journey (CUJ) test coverage strategy for actor message flows in core-term. Each message type between actors represents a CUJ that needs test coverage.

## Current State Analysis

### Coverage Summary

| Component | Current Coverage | Target | Gap |
|-----------|-----------------|--------|-----|
| actor-scheduler framework | 85% | 80% | ✅ Met |
| pixelflow-runtime actors | 45% | 80% | 35% gap |
| core-term actors | 2% | 80% | 78% gap |
| End-to-end flows | 8% | 80% | 72% gap |
| **Overall** | **~35%** | **80%** | **45% gap** |

### Actor Message Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                     MESSAGE FLOW DIAGRAM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  Vec<u8>   ┌─────────────┐  Vec<AnsiCommand>     │
│  │ReadThread├───────────►│ParserThread ├───────────────────┐   │
│  └────┬─────┘  (Data)    └─────────────┘    (SyncSender)   │   │
│       │                                                     │   │
│  ┌────▼─────┐                                               │   │
│  │   PTY    │                                               │   │
│  └────▲─────┘                                               │   │
│       │                                                     ▼   │
│  ┌────┴──────┐  Vec<u8>  ┌─────────────┐   EngineEvent*   ┌───┐│
│  │WriteThread│◄──────────┤TerminalApp  │◄─────────────────┤   ││
│  └───────────┘(SyncSend) └──────┬──────┘   (Control/      │ E ││
│                                 │           Mgmt/Data)    │ n ││
│                                 │                         │ g ││
│                                 │ AppData<P>              │ i ││
│                                 │ (RenderSurface)         │ n ││
│                                 ▼                         │ e ││
│                          ┌──────────────┐                 │   ││
│                          │EngineHandler │◄────────────────┤   ││
│                          └──────┬───────┘  DisplayEvent   └───┘│
│                                 │                              │
│                    EngineControl│AppManagement                 │
│                                 ▼                              │
│                          ┌──────────────┐                      │
│                          │ DriverActor  │                      │
│                          └──────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Message CUJ Inventory

### Priority 1: PTY I/O Chain (Currently 0% → Target 80%)

| CUJ ID | Source | Message | Destination | Priority | Tests Needed |
|--------|--------|---------|-------------|----------|--------------|
| PTY-01 | ReadThread | `Message::Data(Vec<u8>)` | ParserThread | HIGH | 3 |
| PTY-02 | ParserThread | `Vec<AnsiCommand>` | TerminalApp | HIGH | 3 |
| PTY-03 | TerminalApp | `Vec<u8>` | WriteThread | HIGH | 2 |
| PTY-04 | WriteThread | PTY close | ReadThread (cascade) | HIGH | 2 |

**Test scenarios for PTY-01 (ReadThread → ParserThread):**
1. Single byte batch delivery
2. Multiple batches delivered in order (FIFO)
3. Large batch handling (4KB)
4. Channel closure propagation

**Test scenarios for PTY-02 (ParserThread → TerminalApp):**
1. Single command batch delivery
2. Empty input produces no output
3. Partial sequences buffered correctly
4. Mixed commands batched properly

### Priority 2: Engine ↔ App Messages (Currently 15% → Target 80%)

| CUJ ID | Source | Message | Destination | Priority | Tests Needed |
|--------|--------|---------|-------------|----------|--------------|
| ENG-01 | Engine | `EngineEventControl::Resize` | TerminalApp | HIGH | 2 |
| ENG-02 | Engine | `EngineEventControl::CloseRequested` | TerminalApp | HIGH | 1 |
| ENG-03 | Engine | `EngineEventControl::ScaleChanged` | TerminalApp | MED | 1 |
| ENG-04 | Engine | `EngineEventManagement::KeyDown` | TerminalApp | HIGH | 3 |
| ENG-05 | Engine | `EngineEventManagement::MouseClick` | TerminalApp | MED | 2 |
| ENG-06 | Engine | `EngineEventManagement::MouseMove` | TerminalApp | LOW | 1 |
| ENG-07 | Engine | `EngineEventManagement::MouseScroll` | TerminalApp | MED | 1 |
| ENG-08 | Engine | `EngineEventManagement::Paste` | TerminalApp | HIGH | 2 |
| ENG-09 | Engine | `EngineEventData::RequestFrame` | TerminalApp | HIGH | 2 |
| ENG-10 | TerminalApp | `AppData::RenderSurface` | Engine | HIGH | 2 |
| ENG-11 | TerminalApp | `AppData::Skipped` | Engine | MED | 1 |

### Priority 3: VSync Flow (Currently 50% → Target 80%)

| CUJ ID | Source | Message | Destination | Priority | Tests Needed |
|--------|--------|---------|-------------|----------|--------------|
| VSYNC-01 | VsyncActor | `EngineControl::VSync` | Engine | HIGH | 2 |
| VSYNC-02 | Engine | `RenderedResponse` | VsyncActor | HIGH | 1 |
| VSYNC-03 | Engine | `VsyncCommand::Start/Stop` | VsyncActor | MED | 2 |
| VSYNC-04 | Engine | `VsyncCommand::UpdateRefreshRate` | VsyncActor | MED | 1 |

### Priority 4: Display Driver Flow (Currently 55% → Target 80%)

| CUJ ID | Source | Message | Destination | Priority | Tests Needed |
|--------|--------|---------|-------------|----------|--------------|
| DRV-01 | Platform | `DisplayEvent::Key` | Engine | HIGH | 2 |
| DRV-02 | Platform | `DisplayEvent::Resized` | Engine | HIGH | 1 |
| DRV-03 | Platform | `DisplayEvent::CloseRequested` | Engine | HIGH | 1 |
| DRV-04 | Engine | `DriverCommand::Present` | Driver | HIGH | 1 |
| DRV-05 | Engine | `DriverCommand::SetTitle` | Driver | LOW | 1 |

## Execution Plan

### Phase 1: PTY I/O Actor Tests (Week 1)

**File:** `core-term/tests/message_cuj_tests.rs`

```
Step 1.1: Create mock PTY for testing
- MockPty struct implementing necessary traits
- Controllable read/write with test data

Step 1.2: Test ReadThread → ParserThread message flow
- PTY-01: Byte batch delivery tests
- Verify Data lane priority

Step 1.3: Test ParserThread → App message flow
- PTY-02: Command batch delivery tests
- Verify ANSI parsing integration

Step 1.4: Test write path and lifecycle
- PTY-03: Write command delivery
- PTY-04: Cascade shutdown on PTY close
```

### Phase 2: Engine ↔ App Message Tests (Week 2)

**File:** `core-term/tests/engine_app_message_tests.rs`

```
Step 2.1: Test EngineEventControl messages
- ENG-01: Resize triggers terminal resize
- ENG-02: CloseRequested triggers shutdown
- ENG-03: ScaleChanged updates rendering

Step 2.2: Test EngineEventManagement messages
- ENG-04: KeyDown → PTY write
- ENG-05: MouseClick → selection/action
- ENG-08: Paste → PTY write

Step 2.3: Test EngineEventData messages
- ENG-09: RequestFrame → render response

Step 2.4: Test App → Engine responses
- ENG-10: RenderSurface delivery
- ENG-11: Frame skip handling
```

### Phase 3: VSync and Driver Tests (Week 3)

**File:** `pixelflow-runtime/tests/vsync_cuj_tests.rs`

```
Step 3.1: VSync timing message tests
- VSYNC-01: VSync signal delivery
- VSYNC-02: Frame response handling

Step 3.2: VSync control tests
- VSYNC-03: Start/Stop behavior
- VSYNC-04: Refresh rate updates

Step 3.3: Display event routing tests
- DRV-01 through DRV-05
```

### Phase 4: End-to-End Integration Tests (Week 4)

**File:** `core-term/tests/e2e_message_tests.rs`

```
Step 4.1: Full input → output path
- Keyboard input → PTY → parse → render

Step 4.2: Window lifecycle
- Create → resize → close sequence

Step 4.3: Paste workflow
- Engine paste event → PTY write → display
```

## Test Implementation Guidelines

### Test Structure

```rust
#[test]
fn cuj_pty01_single_byte_batch_delivery() {
    // Given: ReadThread with mock PTY containing test data
    // When: PTY returns bytes
    // Then: ParserThread receives exact bytes via Data message
}
```

### Naming Convention

- Test functions: `cuj_{cuj_id}_{scenario_description}`
- Example: `cuj_pty01_single_byte_batch_delivery`
- Example: `cuj_eng04_keydown_with_modifiers`

### Mock Strategy

1. **MockPty**: Fake PTY that returns predetermined data
2. **MockApp**: Collects received messages for verification
3. **TestActorHandle**: Wrapper to capture sent messages

### Assertions

Each test should verify:
1. Message was delivered (not lost)
2. Message content is correct
3. Message arrived at correct destination
4. Message priority lane is correct

## Coverage Metrics

### How to Measure

```bash
# Run with coverage instrumentation
RUSTFLAGS="-C instrument-coverage" cargo test

# Generate coverage report
grcov . -s . --binary-path ./target/debug/ -t html -o ./target/coverage/
```

### Target Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Message types with tests | 8/32 (25%) | 26/32 (80%) |
| Actor pairs tested | 3/8 (37%) | 7/8 (87%) |
| Integration scenarios | 2/10 (20%) | 8/10 (80%) |

## Test File Organization

```
core-term/
├── tests/
│   ├── message_cuj_tests.rs      # PTY I/O chain tests
│   ├── engine_app_message_tests.rs   # Engine ↔ App tests
│   └── e2e_message_tests.rs      # End-to-end tests
│
pixelflow-runtime/
├── tests/
│   ├── vsync_cuj_tests.rs        # VSync message tests
│   └── display_cuj_tests.rs      # Display driver tests
```

## Appendix: Full Message Type Catalog

### actor-scheduler::Message<D, C, M>
- `Data(D)` - Lowest priority, bounded, backpressure
- `Control(C)` - Highest priority, unbounded
- `Management(M)` - Medium priority, unbounded

### EngineEventControl (Engine → App)
- `Resize(u32, u32)`
- `CloseRequested`
- `ScaleChanged(f64)`

### EngineEventManagement (Engine → App)
- `KeyDown { key, mods, text }`
- `MouseClick { x, y, button }`
- `MouseRelease { x, y, button }`
- `MouseMove { x, y, mods }`
- `MouseScroll { x, y, dx, dy, mods }`
- `FocusGained`
- `FocusLost`
- `Paste(String)`

### EngineEventData (Engine → App)
- `RequestFrame { timestamp, target_timestamp, refresh_interval }`

### AppData<P> (App → Engine)
- `RenderSurface(Arc<dyn Manifold>)`
- `RenderSurfaceU32(Arc<dyn Manifold>)`
- `Skipped`

### VsyncCommand
- `Start`
- `Stop`
- `UpdateRefreshRate(f64)`
- `RequestCurrentFPS(Sender<f64>)`
- `Shutdown`

### DisplayEvent
- `WindowCreated`, `WindowDestroyed`
- `Resized`, `ScaleChanged`
- `Key`, `MouseButtonPress/Release`, `MouseMove`, `MouseScroll`
- `FocusGained`, `FocusLost`
- `PasteData`, `ClipboardDataRequested`
- `CloseRequested`
