//! Pod lifecycle types: phase state machine and restart policy.
//!
//! These types correspond to the Kubernetes mental model:
//! - [`PodPhase`] tracks why a scheduler exited (the observable lifecycle state)
//! - [`RestartPolicy`] declares what the supervisor should do on each exit phase

/// The observable phase of a pod (actor thread) at exit.
///
/// Returned by [`ActorScheduler::run`] so a supervisor can decide whether
/// to restart the pod and with what urgency.
///
/// # Phase transitions
///
/// ```text
/// (spawned) ──► Running ──► Terminating ──► Completed   (normal exit)
///                      └──────────────────► Failed(msg) (handler error)
/// ```
///
/// `Pending` is never returned by `run()` — it exists so supervisors can
/// represent a pod that has been declared but not yet started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodPhase {
    /// Declared but not yet started. Not returned by `run()`.
    Pending,

    /// Thread is executing and accepting messages. Not returned by `run()`.
    Running,

    /// Received `Message::Shutdown`, draining per `ShutdownMode`, then exiting.
    /// Not returned by `run()` — transitions directly to `Completed`.
    Terminating,

    /// Clean exit: `Message::Shutdown` received, or all sender handles dropped.
    Completed,

    /// Exited due to `HandlerError::Recoverable`. The message describes the
    /// failure. A supervisor with `RestartPolicy::OnFailure` or `Always`
    /// should respawn the pod.
    ///
    /// `HandlerError::Fatal` is never represented here — it panics immediately.
    Failed(String),
}

impl PodPhase {
    /// Returns `true` if the phase represents an abnormal exit.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, PodPhase::Failed(_))
    }

    /// Returns `true` if the phase represents a clean exit.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(self, PodPhase::Completed)
    }
}

impl std::fmt::Display for PodPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PodPhase::Pending => write!(f, "Pending"),
            PodPhase::Running => write!(f, "Running"),
            PodPhase::Terminating => write!(f, "Terminating"),
            PodPhase::Completed => write!(f, "Completed"),
            PodPhase::Failed(msg) => write!(f, "Failed({msg})"),
        }
    }
}

/// Declares what a supervisor should do when a pod exits.
///
/// Maps directly to Kubernetes restart policy semantics and OTP equivalents.
///
/// | Variant | K8s | OTP |
/// |---------|-----|-----|
/// | `Always` | `Always` | `permanent` |
/// | `OnFailure` | `OnFailure` | `transient` |
/// | `Never` | `Never` | `temporary` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RestartPolicy {
    /// Restart on any exit, including clean `Completed`.
    /// Use for pods that must always be running (display driver, engine).
    Always,

    /// Restart only on `Failed` phase. Do not restart after clean shutdown.
    /// Use for pods that should recover from errors but can be stopped
    /// intentionally (vsync clock, pty monitor).
    #[default]
    OnFailure,

    /// Never restart. Pod runs once and is done.
    /// Use for one-shot setup actors or pods with main-thread constraints
    /// that cannot be re-spawned on an arbitrary thread.
    Never,
}

impl RestartPolicy {
    /// Returns `true` if this policy calls for a restart given the exit phase.
    #[must_use]
    pub fn should_restart(&self, phase: &PodPhase) -> bool {
        match self {
            RestartPolicy::Always => matches!(phase, PodPhase::Completed | PodPhase::Failed(_)),
            RestartPolicy::OnFailure => matches!(phase, PodPhase::Failed(_)),
            RestartPolicy::Never => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_policy_always_restarts_on_both_phases() {
        assert!(RestartPolicy::Always.should_restart(&PodPhase::Completed));
        assert!(RestartPolicy::Always.should_restart(&PodPhase::Failed("oops".into())));
        assert!(!RestartPolicy::Always.should_restart(&PodPhase::Pending));
        assert!(!RestartPolicy::Always.should_restart(&PodPhase::Running));
    }

    #[test]
    fn restart_policy_on_failure_only_restarts_failed() {
        assert!(!RestartPolicy::OnFailure.should_restart(&PodPhase::Completed));
        assert!(RestartPolicy::OnFailure.should_restart(&PodPhase::Failed("oops".into())));
    }

    #[test]
    fn restart_policy_never_never_restarts() {
        assert!(!RestartPolicy::Never.should_restart(&PodPhase::Completed));
        assert!(!RestartPolicy::Never.should_restart(&PodPhase::Failed("oops".into())));
    }

    #[test]
    fn pod_phase_helpers() {
        assert!(PodPhase::Failed("x".into()).is_failed());
        assert!(!PodPhase::Completed.is_failed());
        assert!(PodPhase::Completed.is_completed());
        assert!(!PodPhase::Failed("x".into()).is_completed());
    }

    #[test]
    fn pod_phase_display_formats_each_variant() {
        assert_eq!(PodPhase::Pending.to_string(), "Pending");
        assert_eq!(PodPhase::Running.to_string(), "Running");
        assert_eq!(PodPhase::Terminating.to_string(), "Terminating");
        assert_eq!(PodPhase::Completed.to_string(), "Completed");
        assert_eq!(PodPhase::Failed("boom".into()).to_string(), "Failed(boom)");
    }
}
