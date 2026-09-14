//! Worker process invocation

use std::fmt;

/// How the worker binary should run
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerInvocation {
    /// Credential-free image completeness check
    SelfCheck,
    /// Claim work from a CloudDeck queue
    Claim,
}

/// Invalid worker argv
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationError {
    /// First argument is not `self-check` or `claim`
    UnknownCommand(String),
    /// A subcommand was followed by extra arguments
    UnexpectedArgument,
}

impl fmt::Display for InvocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCommand(command) => {
                write!(f, "unknown worker command {command}")
            }

            Self::UnexpectedArgument => {
                write!(f, "worker command does not take extra arguments")
            }
        }
    }
}

impl std::error::Error for InvocationError {}

/// Selects self-check or claim from argv
///
/// The image ENTRYPOINT is the worker binary. Docker `CMD ["self-check"]` is
/// the credential-free smoke path. Vast `runtype=args` keeps that ENTRYPOINT
/// and appends `WorkerProfile.command`, so the profile command must be `claim`.
/// A missing queue credential must not select self-check.
pub fn worker_invocation<I, S>(args: I) -> Result<WorkerInvocation, InvocationError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let _argv0 = args.next();
    let command = args.next();
    let extra = args.next();
    match (command.as_ref().map(AsRef::as_ref), extra.is_some()) {
        (None, _) => Ok(WorkerInvocation::Claim),
        (Some("self-check"), false) => Ok(WorkerInvocation::SelfCheck),
        (Some("claim"), false) => Ok(WorkerInvocation::Claim),
        (Some(_), true) => Err(InvocationError::UnexpectedArgument),
        (Some(command), false) => Err(InvocationError::UnknownCommand(command.to_owned())),
    }
}
