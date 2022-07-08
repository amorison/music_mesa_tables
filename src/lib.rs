pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod eos_tables;
mod fort_unfmt;
mod index;
mod interp;
mod is_close;
pub mod opacity;
pub mod opacity_tables;
mod raw_tables;
pub mod state;
