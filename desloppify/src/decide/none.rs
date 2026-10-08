//! [`Decide`](super::Decide) that answers nothing.

use anyhow::Result;

use super::{Choice, Decide, Decisions};

pub(super) struct Nobody;

impl Decide for Nobody {
    async fn decide(&self, _state: &str, _questions: &[Choice<'_>]) -> Result<Decisions> {
        Ok(Decisions::default())
    }
}
