// src/ansi/parser.rs

//! ANSI escape sequence parser.
//! Takes individual `AnsiToken`s and accumulates `AnsiCommand`s internally.

use super::batch::AnsiBatch;
use super::commands::{AnsiCommand, C0Control};
use super::lexer::AnsiToken;
use log::{error, trace, warn};
use std::mem;

// Define maximum buffer sizes to prevent excessive memory use
const MAX_PARAMS: usize = 16;
const MAX_INTERMEDIATES: usize = 2;
const MAX_OSC_LEN: usize = 1024; // Limit OSC/DCS/PM/APC string length

/// Represents the current state of the ANSI parser state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Ground,
    Escape,
    CsiEntry,
    CsiParam,
    CsiIntermediate,
    /// Trap state for a CSI sequence that contained a byte outside the
    /// ECMA-48 grammar (e.g. a stray `:` where this parser does not yet
    /// model colon sub-parameters). Absorbs bytes silently until the
    /// terminating final byte, so the malformed sequence's tail is never
    /// misread as printable text. See `dispatch_csi_malformed`.
    CsiIgnore,
    OscString,
    DcsEntry,
    PmString,
    ApcString,
    EscInString,
    EscIntermediate,
}

/// The ANSI parser structure.
///
/// Manages the state transitions and parameter accumulation for parsing
/// ANSI escape sequences.
#[derive(Debug)]
pub struct AnsiParser {
    state: State,
    batch: AnsiBatch,
    params: Vec<u16>,
    intermediates: Vec<u8>,
    string_buffer: Vec<u8>,
    is_private_csi: bool,
    current_param_value: u16,
    is_param_empty: bool,
    string_state_origin: Option<State>,
    esc_intermediate: Option<char>,
}

impl AnsiParser {
    /// Creates a new `AnsiParser`.
    pub fn new() -> Self {
        AnsiParser {
            state: State::Ground,
            batch: AnsiBatch::default(),
            params: Vec::with_capacity(MAX_PARAMS),
            intermediates: Vec::with_capacity(MAX_INTERMEDIATES),
            string_buffer: Vec::with_capacity(MAX_OSC_LEN / 4),
            is_private_csi: false,
            current_param_value: 0,
            is_param_empty: true,
            string_state_origin: None,
            esc_intermediate: None,
        }
    }

    /// Consumes and returns everything parsed so far.
    pub fn take_batch(&mut self) -> AnsiBatch {
        mem::take(&mut self.batch)
    }

    fn clear_csi_state(&mut self) {
        self.params.clear();
        self.intermediates.clear();
        self.is_private_csi = false;
        self.current_param_value = 0;
        self.is_param_empty = true;
    }

    fn clear_string_buffer(&mut self) {
        self.string_buffer.clear();
        self.string_state_origin = None;
    }

    fn clear_esc_state(&mut self) {
        self.esc_intermediate = None;
    }

    fn add_param(&mut self, param: u16) {
        if self.params.len() >= MAX_PARAMS {
            warn!("Exceeded maximum CSI parameters ({})", MAX_PARAMS);
            return;
        }
        self.params.push(param);
    }

    fn finalize_param(&mut self) {
        if self.is_param_empty {
            self.add_param(0);
        } else {
            self.add_param(self.current_param_value);
        }
        self.current_param_value = 0;
        self.is_param_empty = true;
    }

    fn add_intermediate(&mut self, intermediate: u8) {
        if self.intermediates.len() >= MAX_INTERMEDIATES {
            warn!(
                "Exceeded maximum CSI intermediate bytes ({})",
                MAX_INTERMEDIATES
            );
            return;
        }
        self.intermediates.push(intermediate);
    }

    fn add_string_byte(&mut self, byte: u8) {
        if self.string_buffer.len() >= MAX_OSC_LEN {
            warn!("Exceeded maximum string length ({})", MAX_OSC_LEN);
            return;
        }
        self.string_buffer.push(byte);
    }

    fn dispatch_c0(&mut self, byte: u8) {
        trace!("Dispatching C0 Control: {}", byte);
        if let Some(command) = AnsiCommand::from_c0(byte) {
            self.batch.push_command(command);
        } else {
            error!("Unhandled C0 control byte: {}", byte);
            self.batch.push_command(AnsiCommand::Error(byte));
        }
        self.clear_esc_state();
        self.state = State::Ground;
    }

    /// Whether the parser is between sequences, where a printable byte means
    /// only "print this".
    #[inline]
    pub fn is_ground(&self) -> bool {
        self.state == State::Ground
    }

    /// Reserves room for `additional` more bytes of printable text.
    pub fn reserve_text(&mut self, additional: usize) {
        self.batch.reserve_text(additional);
    }

    /// Appends a run of printable ASCII (0x20..=0x7E) to the batch's text.
    ///
    /// Equivalent to feeding each byte as `AnsiToken::Print` in `Ground`: every
    /// byte of the run keeps the parser in `Ground`, so the state machine has
    /// nothing to decide and the run is a single copy.
    pub fn print_ascii_run(&mut self, run: &[u8]) {
        debug_assert!(self.is_ground());
        debug_assert!(run.iter().all(|&b| is_printable_ascii(b)));
        self.clear_esc_state();
        let run = std::str::from_utf8(run).expect("printable ASCII is UTF-8");
        self.batch.push_str(run);
    }

    fn dispatch_print(&mut self, c: char) {
        self.batch.push_char(c);
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn dispatch_csi(&mut self, final_byte: u8) {
        trace!(
            "Dispatching CSI: Private={}, Params={:?}, Intermediates={:?}, Final={}({})",
            self.is_private_csi,
            self.params,
            self.intermediates,
            final_byte as char,
            final_byte
        );
        // Borrow, don't take: taking would leave zero-capacity vectors behind
        // and every following CSI sequence would reallocate both.
        if let Some(command) = AnsiCommand::from_csi(
            &self.params,
            &self.intermediates,
            self.is_private_csi,
            final_byte,
        ) {
            // Check if the command is the specific Unsupported variant we want to remap
            if let AnsiCommand::Csi(super::commands::CsiCommand::Unsupported(
                _,
                Some(unsupported_final_byte),
            )) = command
            {
                trace!(
                    "Remapping CsiCommand::Unsupported with final byte {} to AnsiCommand::Error",
                    unsupported_final_byte
                );
                self.batch
                    .push_command(AnsiCommand::Error(unsupported_final_byte));
            } else {
                // It's a different, valid CSI command
                self.batch.push_command(command);
            }
        } else {
            // AnsiCommand::from_csi returned None, meaning it's not just unsupported but perhaps malformed.
            warn!(
                "AnsiCommand::from_csi returned None for final_byte {}. Reporting as AnsiCommand::Error.",
                final_byte
            );
            self.batch.push_command(AnsiCommand::Error(final_byte));
        }
        self.clear_csi_state();
        self.state = State::Ground;
    }

    fn dispatch_osc(&mut self) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching OSC: Data length {}", data.len());
        self.batch.push_command(AnsiCommand::Osc(data));
        self.clear_string_buffer();
        self.state = State::Ground;
    }

    fn dispatch_dcs(&mut self) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching DCS: Data length {}", data.len());
        self.batch.push_command(AnsiCommand::Dcs(data));
        self.clear_string_buffer();
        self.state = State::Ground;
    }

    fn dispatch_pm(&mut self) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching PM: Data length {}", data.len());
        self.batch.push_command(AnsiCommand::Pm(data));
        self.clear_string_buffer();
        self.state = State::Ground;
    }

    fn dispatch_apc(&mut self) {
        let data = mem::take(&mut self.string_buffer);
        trace!("Dispatching APC: Data length {}", data.len());
        self.batch.push_command(AnsiCommand::Apc(data));
        self.clear_string_buffer();
        self.state = State::Ground;
    }

    fn dispatch_st_standalone(&mut self) {
        trace!("Dispatching Standalone String Terminator (ST)");
        self.batch.push_command(AnsiCommand::StringTerminator);
        self.clear_string_buffer();
        self.clear_esc_state();
        self.state = State::Ground;
    }

    fn dispatch_ignore(&mut self, byte: u8) {
        trace!("Dispatching Ignore: {}", byte);
        self.batch.push_command(AnsiCommand::Ignore(byte));
    }

    fn dispatch_error(&mut self, byte: u8) {
        trace!("Dispatching Error: {}", byte);
        self.batch.push_command(AnsiCommand::Error(byte));
        self.clear_esc_state();
        self.state = State::Ground;
    }

    /// Reports a byte that is not valid anywhere in the CSI grammar (digit,
    /// `;`, private marker, intermediate, or final byte) and traps the
    /// parser in `CsiIgnore` for the remainder of the sequence.
    ///
    /// Falling straight back to `Ground` here (as `dispatch_error` does) would
    /// leave the rest of the malformed sequence's bytes to be read as literal
    /// printable text — a silent fall-through into the wrong state. Instead
    /// the sequence's tail is discarded up to its final byte, matching how
    /// the ECMA-48 "CSI ignore" state recovers, and no command is dispatched
    /// for the discarded final byte since the error was already reported here.
    fn dispatch_csi_malformed(&mut self, byte: u8) {
        warn!(
            "Malformed CSI sequence: unexpected byte {} ({})",
            byte, byte as char
        );
        self.batch.push_command(AnsiCommand::Error(byte));
        self.clear_csi_state();
        self.state = State::CsiIgnore;
    }

    fn enter_string_state(&mut self, next_state: State) {
        self.clear_string_buffer();
        self.string_state_origin = Some(self.state.clone());
        self.state = next_state;
    }

    fn enter_esc_in_string_state(&mut self) {
        self.string_state_origin = Some(self.state.clone());
        self.state = State::EscInString;
    }

    /// Feeds a token into the parser state machine.
    ///
    /// # Parameters
    /// * `token` - The ANSI token to process.
    pub fn process_token(&mut self, token: AnsiToken) {
        match self.state {
            State::Ground => match token {
                AnsiToken::Print(c) => self.dispatch_print(c),
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => self.dispatch_c0(byte),
            },
            State::Escape => match token {
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => self.state = State::Escape,
                AnsiToken::Print('[') => {
                    self.clear_csi_state();
                    self.state = State::CsiEntry;
                }
                AnsiToken::Print(']') => self.enter_string_state(State::OscString),
                AnsiToken::Print('P') => self.enter_string_state(State::DcsEntry),
                AnsiToken::Print('^') => self.enter_string_state(State::PmString),
                AnsiToken::Print('_') => self.enter_string_state(State::ApcString),
                AnsiToken::Print('k') => self.enter_string_state(State::ApcString),
                AnsiToken::Print('\\') => self.dispatch_st_standalone(),
                AnsiToken::Print(inter @ ('(' | ')' | '*' | '+')) => {
                    self.esc_intermediate = Some(inter);
                    self.state = State::EscIntermediate;
                }
                AnsiToken::Print(c) => {
                    if let Some(command) = AnsiCommand::from_esc(c) {
                        // AnsiCommand::from_esc is from commands.rs
                        self.batch.push_command(command);
                    } else {
                        // If 'c' does not form a valid ESC sequence, treat 'c' as a printable character.
                        self.batch.push_char(c);
                    }
                    self.state = State::Ground;
                }
                AnsiToken::C0Control(byte) => self.dispatch_c0(byte),
            },
            State::EscIntermediate => {
                if let Some(inter) = self.esc_intermediate {
                    match token {
                        AnsiToken::Print(final_char) => {
                            // Use the specific helper for ESC intermediate sequences
                            if let Some(command) =
                                AnsiCommand::from_esc_intermediate(inter, final_char)
                            {
                                self.batch.push_command(command);
                            } else {
                                self.dispatch_ignore(inter as u8);
                                self.dispatch_ignore(final_char as u8);
                            }
                        }
                        AnsiToken::C0Control(byte) => {
                            self.dispatch_ignore(inter as u8);
                            self.dispatch_c0(byte);
                        }
                    }
                } else {
                    error!("Invalid EscIntermediate state");
                    self.dispatch_error(token.to_byte_lossy());
                }
                self.clear_esc_state();
                self.state = State::Ground;
            }

            State::CsiEntry => match token {
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(c @ '0'..='9') => {
                    self.current_param_value = (c as u16) - ('0' as u16);
                    self.is_param_empty = false;
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(';') => {
                    self.add_param(0);
                    self.is_param_empty = true;
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(p @ ('?' | '>' | '!' | '$' | '\'')) => {
                    self.is_private_csi = true;
                    self.add_intermediate(p as u8);
                    self.state = State::CsiParam;
                }
                AnsiToken::Print(i @ ' '..='/') => {
                    self.add_intermediate(i as u8);
                    self.state = State::CsiIntermediate;
                }
                AnsiToken::Print(f @ '@'..='~') => self.dispatch_csi(f as u8),
                AnsiToken::Print(c) => self.dispatch_csi_malformed(c as u8),
            },
            State::CsiParam => match token {
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(c @ '0'..='9') => {
                    if let Some(next_val) = self.current_param_value.checked_mul(10) {
                        if let Some(final_val) = next_val.checked_add((c as u16) - ('0' as u16)) {
                            self.current_param_value = final_val;
                        } else {
                            warn!("CSI param overflow");
                            self.current_param_value = u16::MAX;
                        }
                    } else {
                        warn!("CSI param overflow");
                        self.current_param_value = u16::MAX;
                    }
                    self.is_param_empty = false;
                }
                AnsiToken::Print(';') => self.finalize_param(),
                AnsiToken::Print(i @ ' '..='/') => {
                    self.finalize_param();
                    self.add_intermediate(i as u8);
                    self.state = State::CsiIntermediate;
                }
                AnsiToken::Print(f @ '@'..='~') => {
                    self.finalize_param();
                    self.dispatch_csi(f as u8);
                }
                AnsiToken::Print(c) => self.dispatch_csi_malformed(c as u8),
            },
            State::CsiIntermediate => match token {
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                AnsiToken::Print(i @ ' '..='/') => self.add_intermediate(i as u8),
                AnsiToken::Print(f @ '@'..='~') => self.dispatch_csi(f as u8),
                AnsiToken::Print(c) => self.dispatch_csi_malformed(c as u8),
            },
            State::CsiIgnore => match token {
                AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                    self.clear_csi_state();
                    self.clear_esc_state();
                    self.state = State::Escape;
                }
                AnsiToken::C0Control(byte) => {
                    self.dispatch_c0(byte);
                    self.clear_csi_state();
                }
                // The final byte ends the malformed sequence. Its `Error` was
                // already reported when the sequence was first found invalid,
                // so nothing further is dispatched here — just recover.
                AnsiToken::Print(_f @ '@'..='~') => {
                    self.clear_csi_state();
                    self.state = State::Ground;
                }
                // Any other byte (digits, `;`, `:`, private markers,
                // intermediates, or anything else) is silently absorbed:
                // this state's entire purpose is to discard the tail of a
                // sequence already known to be malformed.
                AnsiToken::Print(_) => {}
            },
            State::OscString | State::DcsEntry | State::PmString | State::ApcString => {
                match token {
                    AnsiToken::C0Control(b) if b == C0Control::ESC as u8 => {
                        self.enter_esc_in_string_state()
                    }
                    AnsiToken::C0Control(b)
                        if b == C0Control::BEL as u8 && self.state == State::OscString =>
                    {
                        self.dispatch_osc()
                    }
                    AnsiToken::C0Control(b)
                        if b == C0Control::CAN as u8 || b == C0Control::SUB as u8 =>
                    {
                        self.clear_string_buffer();
                        self.state = State::Ground;
                    }
                    AnsiToken::C0Control(byte) => self.add_string_byte(byte),
                    AnsiToken::Print(c) => {
                        let mut buf = [0; 4];
                        c.encode_utf8(&mut buf)
                            .as_bytes()
                            .iter()
                            .for_each(|&b| self.add_string_byte(b));
                    }
                }
            }
            State::EscInString => match token {
                AnsiToken::Print('\\') => match self.string_state_origin {
                    Some(State::OscString) => self.dispatch_osc(),
                    Some(State::DcsEntry) => self.dispatch_dcs(),
                    Some(State::PmString) => self.dispatch_pm(),
                    Some(State::ApcString) => self.dispatch_apc(),
                    _ => {
                        error!("EscInString state missing origin!");
                        self.dispatch_st_standalone();
                    }
                },
                _ => {
                    trace!(
                        "ESC followed by {:?} inside string - aborting string, processing ESC sequence",
                        token
                    );
                    self.clear_string_buffer();
                    self.batch
                        .push_command(AnsiCommand::C0Control(C0Control::ESC));
                    self.state = State::Ground;
                    self.process_token(token);
                }
            },
        }
    }
}

/// A byte that, in `Ground` with no UTF-8 sequence pending, is exactly
/// `Print(byte as char)`: printable ASCII, excluding every C0 control and DEL.
#[inline]
pub(super) fn is_printable_ascii(byte: u8) -> bool {
    (b' '..=b'~').contains(&byte)
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}

trait AnsiTokenByte {
    fn to_byte_lossy(&self) -> u8;
}

impl AnsiTokenByte for AnsiToken {
    fn to_byte_lossy(&self) -> u8 {
        match self {
            AnsiToken::Print(c) => (*c as u32).try_into().unwrap_or(b'?'),
            AnsiToken::C0Control(b) => *b,
        }
    }
}
