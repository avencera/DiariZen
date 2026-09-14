//! `open-yap-review serve`: the private localhost Open Yap review server

use std::{
    io::Write,
    net::{Ipv4Addr, SocketAddr, TcpListener},
    path::PathBuf,
    sync::Arc,
};

use clap::{Parser, Subcommand};
use color_eyre::eyre::{Result, WrapErr};
use open_yap_review::{
    http::{self, Access, AppState, Role, UiDist, ui::default_dist_dir},
    store::ReviewStore,
    text::NonEmptyText,
};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "open-yap-review", about = "Private Open Yap review server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the review UI and API on 127.0.0.1
    Serve(ServeArgs),
}

#[derive(Debug, clap::Args)]
struct ServeArgs {
    /// Review session directory created by `open_yap_review.py prepare`
    #[arg(long)]
    session: PathBuf,

    /// Actor id recorded on every event
    #[arg(long)]
    reviewer_id: String,

    /// Reject all POST requests
    #[arg(long)]
    read_only: bool,

    /// Accept sign-off decisions instead of reviewer events
    #[arg(long)]
    signoff: bool,

    /// Port on 127.0.0.1, 0 picks a free port
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// Built UI directory, defaults to recipes/speakrs/review-ui/dist in the repository
    #[arg(long)]
    ui_dist: Option<PathBuf>,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    let Command::Serve(args) = Cli::parse().command;
    serve(args)
}

fn serve(args: ServeArgs) -> Result<()> {
    let actor = NonEmptyText::parse_str(&args.reviewer_id, "reviewer_id")?;
    let ui_dist = args.ui_dist.unwrap_or_else(default_dist_dir);
    let ui = UiDist::open(&ui_dist)?;
    let store = ReviewStore::open(&args.session)
        .wrap_err_with(|| format!("failed to open review session {}", args.session.display()))?;
    let access = if args.read_only {
        Access::ReadOnly
    } else {
        Access::ReadWrite
    };
    let role = if args.signoff {
        Role::Signer
    } else {
        Role::Reviewer
    };
    let state = Arc::new(AppState::new(store, actor, access, role, ui));

    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, args.port)))
        .wrap_err("failed to bind 127.0.0.1")?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();

    let ready = json!({
        "ok": true,
        "host": "127.0.0.1",
        "port": port,
        "url": format!("http://127.0.0.1:{port}/"),
        "read_only": args.read_only,
        "signoff_mode": args.signoff,
        "reviewer_id": args.reviewer_id,
    });
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{ready}")?;
    stdout.flush()?;
    drop(stdout);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::from_std(listener)?;
        http::serve(listener, state, shutdown_signal()).await?;
        Ok(())
    })
}

async fn shutdown_signal() {
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!("ctrl-c handler failed error={error}");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!("SIGTERM handler failed error={error}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}
