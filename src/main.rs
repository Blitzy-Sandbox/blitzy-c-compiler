// Minimal entry point for bcc (Blitzy C Compiler)
// This stub enables compilation and testing of individual modules during
// the greenfield build-out. It will be replaced with the full driver
// implementation by its assigned agent.

mod common;
mod frontend;
mod sema;

fn main() {
    // Full CLI driver implementation pending.
    // This stub exists solely to enable `cargo build` and `cargo test`
    // for the common module during development.
    std::process::exit(0);
}
