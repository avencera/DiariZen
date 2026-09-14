//! Python evaluator process-group lifecycle

use std::{
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc::{self, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;
use thiserror::Error;

use crate::documents::MAX_RESULT_BYTES;

const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(50);
const PROCESS_GROUP_GRACE: Duration = Duration::from_millis(500);
const PREREQUISITE_TIMEOUT: Duration = Duration::from_secs(60);

/// Evaluator command and argument vector
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorCommand {
    program: String,
    args: Vec<String>,
}

impl EvaluatorCommand {
    /// Builds a command that reads one request file
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Result<Self, ProcessError> {
        let program = program.into();
        if program.is_empty() || args.iter().any(|argument| argument.contains('\0')) {
            return Err(ProcessError::InvalidCommand);
        }
        Ok(Self { program, args })
    }

    /// Default in-image evaluator invocation
    #[must_use]
    pub fn default_python_module() -> Self {
        Self {
            program: "python3.10".to_owned(),
            args: vec![
                "-m".to_owned(),
                "recipes.speakrs.large.validation.evaluator".to_owned(),
            ],
        }
    }

    /// Program plus arguments without the request-file flag
    #[must_use]
    pub fn argv(&self) -> Vec<String> {
        let mut argv = vec![self.program.clone()];
        argv.extend(self.args.iter().cloned());
        argv
    }
}

/// Bounded evaluator stdout bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorOutput(Vec<u8>);

impl EvaluatorOutput {
    /// Returns the bounded evaluator stdout bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Evaluator process failure
#[derive(Debug, Error)]
pub enum ProcessError {
    /// Command identity was empty or contained a NUL
    #[error("evaluator command is invalid")]
    InvalidCommand,
    /// The process could not be spawned
    #[error("evaluator process could not start")]
    Spawn,
    /// The evaluator failed or stdout could not be captured within the result limit
    #[error("evaluator command or stdout failed")]
    Output,
    /// The process was cancelled or killed
    #[error("evaluator process was cancelled")]
    Cancelled,
    /// A completion arrived after cancellation
    #[error("evaluator result arrived after cancellation")]
    LateResult,
}

/// Runs one evaluator in a new process group and bounds its stdout
pub fn run_evaluator(
    command: &EvaluatorCommand,
    request_file: &Path,
    cancelled: impl Fn() -> bool,
) -> Result<EvaluatorOutput, ProcessError> {
    if cancelled() {
        return Err(ProcessError::Cancelled);
    }

    let mut child = {
        let mut process = Command::new(&command.program);
        process
            .args(&command.args)
            .arg("--request-file")
            .arg(request_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null());
        isolate_process_group(&mut process);
        process.spawn().map_err(|_| ProcessError::Spawn)?
    };

    let stdout = child.stdout.take().ok_or(ProcessError::Spawn)?;
    let (output_sender, output_receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let _ = output_sender.send(capture_stdout(stdout));
    });

    let deadline = Instant::now() + Duration::from_secs(60 * 60);
    let mut captured = None;
    loop {
        if cancelled() {
            terminate_group(&mut child);
            let _ = reader.join();
            return Err(ProcessError::Cancelled);
        }

        if captured.is_none() {
            match output_receiver.try_recv() {
                Ok(CapturedStdout::Complete(bytes)) => captured = Some(bytes),
                Ok(CapturedStdout::Overflow | CapturedStdout::Failed) => {
                    terminate_group(&mut child);
                    let _ = reader.join();
                    return Err(ProcessError::Output);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    terminate_group(&mut child);
                    let _ = reader.join();
                    return Err(ProcessError::Output);
                }
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                terminate_group(&mut child);
                let bytes = match captured {
                    Some(bytes) => bytes,
                    None => match output_receiver.recv_timeout(PROCESS_GROUP_GRACE) {
                        Ok(CapturedStdout::Complete(bytes)) => bytes,
                        Ok(CapturedStdout::Overflow | CapturedStdout::Failed) | Err(_) => {
                            let _ = reader.join();
                            return Err(ProcessError::Output);
                        }
                    },
                };
                reader.join().map_err(|_| ProcessError::Output)?;

                if cancelled() {
                    return Err(ProcessError::LateResult);
                }
                if !status.success() {
                    return Err(ProcessError::Output);
                }
                return Ok(EvaluatorOutput(bytes));
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    terminate_group(&mut child);
                    let _ = reader.join();
                    return Err(ProcessError::Output);
                }
                thread::sleep(PROCESS_POLL_INTERVAL);
            }
            Err(_) => {
                terminate_group(&mut child);
                let _ = reader.join();
                return Err(ProcessError::Spawn);
            }
        }
    }
}

/// Checks that the configured evaluator can load without credentials or work input
pub fn check_evaluator_prerequisite(command: &EvaluatorCommand) -> Result<(), ProcessError> {
    let mut process = Command::new(&command.program);
    process
        .args(&command.args)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null());
    isolate_process_group(&mut process);

    let mut child = process.spawn().map_err(|_| ProcessError::Spawn)?;
    let deadline = Instant::now() + PREREQUISITE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                terminate_group(&mut child);

                return status.success().then_some(()).ok_or(ProcessError::Output);
            }
            Ok(None) if Instant::now() < deadline => thread::sleep(PROCESS_POLL_INTERVAL),
            Ok(None) => {
                terminate_group(&mut child);

                return Err(ProcessError::Output);
            }
            Err(_) => {
                terminate_group(&mut child);

                return Err(ProcessError::Spawn);
            }
        }
    }
}

#[cfg(unix)]
fn isolate_process_group(process: &mut Command) {
    use std::os::unix::process::CommandExt;

    process.process_group(0);
}

#[cfg(not(unix))]
fn isolate_process_group(_: &mut Command) {}

#[derive(Debug)]
enum CapturedStdout {
    Complete(Vec<u8>),
    Overflow,
    Failed,
}

fn capture_stdout(mut stdout: impl Read) -> CapturedStdout {
    let mut bytes = Vec::with_capacity((MAX_RESULT_BYTES as usize).min(8 * 1024));
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = match stdout.read(&mut buffer) {
            Ok(read) => read,
            Err(_) => return CapturedStdout::Failed,
        };
        if read == 0 {
            return CapturedStdout::Complete(bytes);
        }
        if bytes.len() as u64 + read as u64 > u64::from(MAX_RESULT_BYTES) {
            return CapturedStdout::Overflow;
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
}

fn terminate_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }

        let wait_until = Instant::now() + PROCESS_GROUP_GRACE;
        while Instant::now() < wait_until {
            let child_finished = child.try_wait().ok().flatten().is_some();
            if child_finished && !process_group_exists(pid) {
                return;
            }
            thread::sleep(PROCESS_POLL_INTERVAL);
        }

        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        let _ = child.wait();

        let wait_until = Instant::now() + PROCESS_GROUP_GRACE;
        while Instant::now() < wait_until && process_group_exists(pid) {
            thread::sleep(PROCESS_POLL_INTERVAL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(unix)]
fn process_group_exists(process_group: i32) -> bool {
    let result = unsafe { libc::kill(-process_group, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Writes a request document to a local file for the evaluator
pub fn write_request_file(path: &Path, request: &impl Serialize) -> Result<(), ProcessError> {
    let encoded = serde_json::to_vec(request).map_err(|_| ProcessError::Output)?;
    let mut file = std::fs::File::create(path).map_err(|_| ProcessError::Spawn)?;
    file.write_all(&encoded).map_err(|_| ProcessError::Spawn)?;
    Ok(())
}
