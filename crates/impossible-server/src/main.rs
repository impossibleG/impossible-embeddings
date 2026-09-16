//! Impossible Embedding process entry point.

use clap::Parser;
use impossible_server::cli::{Cli, run};
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    run(Cli::parse()).await
}
